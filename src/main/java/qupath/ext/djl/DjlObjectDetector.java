package qupath.ext.djl;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.ROIs;

/**
 * Helper class for object detection using Deep Java Library with custom translators.
 * 
 * @author Your Name
 */
public class DjlObjectDetector implements AutoCloseable {
    
    private static final Logger logger = LoggerFactory.getLogger(DjlObjectDetector.class);
    
    private ZooModel<Image, DetectedObjects> model;
    private String engine;
    private URI modelUri;
    private Translator<Image, DetectedObjects> translator;
    private int inputSize;
    
    /**
     * Create a new DjlObjectDetector.
     * 
     * @param engine the DJL engine to use (e.g. "PyTorch", "TensorFlow")
     * @param modelUri the URI to the model file
     * @param translator the custom translator to use for the model
     * @param inputSize the expected input size for the model (e.g. 640 for YOLOv8)
     * @throws ModelNotFoundException
     * @throws MalformedModelException
     * @throws IOException
     */
    public DjlObjectDetector(String engine, URI modelUri, Translator<Image, DetectedObjects> translator, int inputSize) 
            throws ModelNotFoundException, MalformedModelException, IOException {
        this.engine = engine;
        this.modelUri = modelUri;
        this.translator = translator;
        this.inputSize = inputSize;
        initializeModel();
    }
    
    private void initializeModel() throws ModelNotFoundException, MalformedModelException, IOException {
        if (model != null)
            return;
            
        model = DjlTools.loadModel(
            engine,
            Image.class,
            DetectedObjects.class,
            translator,
            modelUri
        );
        
        logger.info("Model loaded successfully from {}", modelUri);
    }
    
    private static List<PathObject> detectionsToPathObjects(DetectedObjects detections, RegionRequest request, int inputSize) {
        List<PathObject> objects = new ArrayList<>();
        double scale = request.getWidth() / (double)inputSize;
    
        for (var detection : detections.items()) {
            var bbox = ((DetectedObject) detection).getBoundingBox();
            var bounds = bbox.getBounds();
            
            double centerX = bounds.getX() * scale + request.getX();
            double centerY = bounds.getY() * scale + request.getY();
            double width = bounds.getWidth() * scale;
            double height = bounds.getHeight() * scale;
            
            var roi = ROIs.createRectangleROI(
                centerX,
                centerY,
                width,
                height,
                request.getImagePlane()
            );
        
            var pathClass = PathClass.fromString(detection.getClassName());
            var annotation = PathObjects.createAnnotationObject(roi, pathClass);
            try {
                annotation.getMeasurementList().put("Class probability", detection.getProbability());
            } catch (Exception e) {
                logger.warn("Unable to add probability measurement: " + e.getLocalizedMessage(), e);
            }
            objects.add(annotation);
        }
        return objects;
    }

    /**
     * Detect objects across an entire image.
     * 
     * @param imageData the image data to process
     * @return Optional containing list of detected objects, or empty if detection was interrupted
     * @throws TranslateException
     * @throws IOException
     */
    public Optional<List<PathObject>> detect(ImageData<BufferedImage> imageData) throws TranslateException, IOException {
        return detect(imageData, Collections.singleton(imageData.getHierarchy().getRootObject()));
    }

    /**
     * Detect objects within specified parent objects in an image.
     * 
     * @param imageData the image data to process
     * @param parentObjects the parent objects within which to detect
     * @return Optional containing list of detected objects, or empty if detection was interrupted
     * @throws TranslateException
     * @throws IOException
     */
    public Optional<List<PathObject>> detect(ImageData<BufferedImage> imageData, Collection<? extends PathObject> parentObjects) throws TranslateException, IOException {
        if (parentObjects == null)
            parentObjects = Collections.singleton(imageData.getHierarchy().getRootObject());

        var server = imageData.getServer();
        double downsampleBase = server.getDownsampleForResolution(0);
        
        // Map to store parent-child relationships
        var map = new ConcurrentHashMap<PathObject, List<PathObject>>();
        // List of all objects created
        var list = new ArrayList<PathObject>();
        
        try (var predictor = model.newPredictor()) {
            for (var parent : parentObjects) {
                parent.clearChildObjects();
                
                // Create region request(s)
                List<RegionRequest> requests;
                var roi = parent.getROI();
                if (roi == null) {
                    requests = Collections.singletonList(
                            RegionRequest.createInstance(server.getPath(), downsampleBase, 
                                    0, 0, server.getWidth(), server.getHeight())
                    );
                } else {
                    requests = Collections.singletonList(
                            RegionRequest.createInstance(server.getPath(), downsampleBase, roi)
                    );
                }
                
                var childObjects = map.computeIfAbsent(parent, p -> new ArrayList<>());
                for (var request : requests) {
                    if (Thread.currentThread().isInterrupted()) {
                        logger.warn("Detection interrupted! Discarding {} detection(s)", list.size());
                        return Optional.empty();
                    }
                    
                    var img = server.readRegion(request);
                    var djlImage = ImageFactory.getInstance().fromImage(img);
                    var detections = predictor.predict(djlImage);
                    
                    var detected = detectionsToPathObjects(detections, request, inputSize);
                    childObjects.addAll(detected);
                    list.addAll(detected);
                }
            }
        }
        
        // Update hierarchy
        for (var entry : map.entrySet()) {
            var parent = entry.getKey();
            var childObjects = entry.getValue();
            parent.clearChildObjects();
            parent.addChildObjects(childObjects);
        }
        
        imageData.getHierarchy().fireHierarchyChangedEvent(this);
        return Optional.of(list);
    }

    @Override
    public void close() throws Exception {
        if (model != null) {
            model.close();
            model = null;
        }
    }
}