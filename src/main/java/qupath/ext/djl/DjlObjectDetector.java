package qupath.ext.djl;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

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
    private double overlapPercentage;
    
    /**
     * Create a new DjlObjectDetector.
     * 
     * @param engine the DJL engine to use (e.g. "PyTorch", "TensorFlow")
     * @param modelUri the URI to the model file
     * @param translator the custom translator to use for the model
     * @param inputSize the expected input size for the model (e.g. 640 for YOLOv8)
     * @param overlapPercentage the percentage of overlap between tiles (0.0 to 1.0)
     * @throws ModelNotFoundException
     * @throws MalformedModelException
     * @throws IOException
     */
    public DjlObjectDetector(String engine, URI modelUri, Translator<Image, DetectedObjects> translator, int inputSize, double overlapPercentage) 
            throws ModelNotFoundException, MalformedModelException, IOException {
        this.engine = engine;
        this.modelUri = modelUri;
        this.translator = translator;
        this.inputSize = inputSize;
        this.overlapPercentage = Math.min(Math.max(overlapPercentage, 0.0), 1.0);
        initializeModel();
        System.setProperty("logging.level.qupath.ext.djl", "DEBUG");
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


    private List<RegionRequest> createTiledRequests(String imagePath, double downsample, int x, int y, int width, int height) {
        List<RegionRequest> requests = new ArrayList<>();
        
        int stride = (int) (inputSize * (1 - overlapPercentage));
        
        for (int tileX = x; tileX < x + width; tileX += stride) {
            for (int tileY = y; tileY < y + height; tileY += stride) {
                int tileWidth = Math.min(inputSize, x + width - tileX);
                int tileHeight = Math.min(inputSize, y + height - tileY);
                
                RegionRequest request = RegionRequest.createInstance(
                    imagePath, downsample, tileX, tileY, tileWidth, tileHeight);
                requests.add(request);
            }
        }
        
        return requests;
    }

    private boolean doObjectsOverlap(PathObject obj1, PathObject obj2) {
        // Get ROIs in global image coordinates
        qupath.lib.roi.interfaces.ROI roi1 = obj1.getROI();
        qupath.lib.roi.interfaces.ROI roi2 = obj2.getROI();
        
        // First do a quick bounds check (optimization)
        if (!roi1.getGeometry().getBoundary().intersects(roi2.getGeometry().getBoundary())) {
            return false;
        }
        
        // Then do precise geometry intersection
        return roi1.getGeometry().intersects(roi2.getGeometry());
    }

    private List<PathObject> mergeOverlappingDetections(List<PathObject> detections) {
        List<PathObject> merged = new ArrayList<>();
        boolean[] processed = new boolean[detections.size()];
        
        // Sort detections by probability to prioritize high-confidence detections
        List<PathObject> sortedDetections = new ArrayList<>(detections);
        sortedDetections.sort((o1, o2) -> Double.compare(
            o2.getMeasurementList().get("Class probability"),
            o1.getMeasurementList().get("Class probability")));
        
        for (int i = 0; i < sortedDetections.size(); i++) {
            if (processed[i]) continue;
            
            PathObject current = sortedDetections.get(i);
            processed[i] = true;
            
            // Find all overlapping objects in global image coordinates
            List<PathObject> overlapping = new ArrayList<>();
            for (int j = i + 1; j < sortedDetections.size(); j++) {
                if (!processed[j]) {
                    PathObject other = sortedDetections.get(j);
                    // Check overlap in global image coordinates
                    if (doObjectsOverlap(current, other)) {
                        overlapping.add(other);
                        processed[j] = true;
                    }
                }
            }
            
            if (overlapping.isEmpty()) {
                merged.add(current);
            } else {
                // Keep the detection with highest probability
                PathObject best = current;
                double maxProb = current.getMeasurementList().get("Class probability");
                
                for (PathObject obj : overlapping) {
                    double prob = obj.getMeasurementList().get("Class probability");
                    if (prob > maxProb) {
                        maxProb = prob;
                        best = obj;
                    }
                }
                merged.add(best);
            }
        }
        
        return merged;
    }

    private List<PathObject> applyNMS(List<DetectedObject> detections, RegionRequest request) {
        final double NMS_THRESHOLD = 0.45;
        List<PathObject> result = new ArrayList<>();
        
        // Group detections by class
        var detectionsByClass = detections.stream()
            .collect(Collectors.groupingBy(DetectedObject::getClassName));
        
        // Apply NMS for each class separately
        for (var entry : detectionsByClass.entrySet()) {
            var classDetections = entry.getValue();
            
            // Sort by confidence
            classDetections.sort((a, b) -> Double.compare(b.getProbability(), a.getProbability()));
            
            List<DetectedObject> kept = new ArrayList<>();
            boolean[] suppressed = new boolean[classDetections.size()];
            
            for (int i = 0; i < classDetections.size(); i++) {
                if (suppressed[i]) continue;
                
                kept.add(classDetections.get(i));
                
                // Suppress lower confidence detections
                for (int j = i + 1; j < classDetections.size(); j++) {
                    if (!suppressed[j]) {
                        var box1 = classDetections.get(i).getBoundingBox();
                        var box2 = classDetections.get(j).getBoundingBox();
                        
                        double iou = box1.getIoU(box2);
                        logger.debug("Box1 prob: {:.3f}, Box2 prob: {:.3f}, IoU: {:.3f}", 
                        classDetections.get(i).getProbability(),
                        classDetections.get(j).getProbability(),
                        iou);
                        if (iou > NMS_THRESHOLD) {
                            suppressed[j] = true;
                        }
                    }
                }
            }
            
            // Convert kept detections to PathObjects
            for (var detection : kept) {
                var bbox = detection.getBoundingBox();
                var bounds = bbox.getBounds();
                
                double scale = request.getWidth() / (double)inputSize;
                double centerX = bounds.getX() * scale + request.getX();
                double centerY = bounds.getY() * scale + request.getY();
                double width = bounds.getWidth() * scale;
                double height = bounds.getHeight() * scale;
                
                var roi = ROIs.createRectangleROI(
                    centerX, centerY, width, height,
                    request.getImagePlane()
                );
                
                var pathClass = PathClass.fromString(detection.getClassName());
                var annotation = PathObjects.createAnnotationObject(roi, pathClass);
                annotation.getMeasurementList().put("Class probability", detection.getProbability());
                result.add(annotation);
            }
        }
        
        return result;
    }

    private boolean isRegionOverlapping(RegionRequest request, Set<RegionRequest> processedRegions) {
        for (RegionRequest processed : processedRegions) {
            if (request.getX() >= processed.getX() && 
                request.getY() >= processed.getY() &&
                request.getX() + request.getWidth() <= processed.getX() + processed.getWidth() &&
                request.getY() + request.getHeight() <= processed.getY() + processed.getHeight()) {
                return true;
            }
        }
        return false;
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
        
        var map = new ConcurrentHashMap<PathObject, List<PathObject>>();
        var list = new ArrayList<PathObject>();
        Set<RegionRequest> processedRegions = ConcurrentHashMap.newKeySet();
        
        for (var parent : parentObjects) {
            parent.clearChildObjects();
                
            List<RegionRequest> requests;
            var roi = parent.getROI();
            if (roi == null) {
                throw new IllegalArgumentException("ROI cannot be null");
            }
            
            requests = createTiledRequests(server.getPath(), downsampleBase,
                    (int)roi.getBoundsX(), (int)roi.getBoundsY(), 
                    (int)roi.getBoundsWidth(), (int)roi.getBoundsHeight());

            var childObjects = map.computeIfAbsent(parent, p -> new ArrayList<>());
            List<PathObject> allTileDetections = new ArrayList<>();
                
            // Process each tile individually
            for (var request : requests) {
                if (isRegionOverlapping(request, processedRegions)) {
                    logger.debug("Skipping overlapping region at {},{}", request.getX(), request.getY());
                    continue;
                }
                processedRegions.add(request);
                
                if (Thread.currentThread().isInterrupted()) {
                    logger.warn("Detection interrupted! Discarding {} detection(s)", list.size());
                    return Optional.empty();
                }
                    
                var img = server.readRegion(request);
                logger.debug("Tile dimensions before resize: {}x{}", img.getWidth(), img.getHeight());
                var djlImage = ImageFactory.getInstance().fromImage(img).resize(inputSize, inputSize, true);
                try(var predictor = model.newPredictor()) {
                    var detections = predictor.predict(djlImage);
                    // Apply NMS directly to raw detections
                    allTileDetections.addAll(applyNMS(detections.items(), request));
                }
            }
                
            // Merge overlapping detections
            var mergedDetections = mergeOverlappingDetections(allTileDetections);
            childObjects.addAll(mergedDetections);
            list.addAll(mergedDetections);
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