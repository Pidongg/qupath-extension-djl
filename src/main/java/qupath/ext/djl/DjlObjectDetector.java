package qupath.ext.djl;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
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
    private double nmsThreshold;
    
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
        this(engine, modelUri, translator, inputSize, overlapPercentage, 0.45);
    }
    
    /**
     * Create a new DjlObjectDetector with custom NMS threshold.
     * 
     * @param engine the DJL engine to use (e.g. "PyTorch", "TensorFlow")
     * @param modelUri the URI to the model file
     * @param translator the custom translator to use for the model
     * @param inputSize the expected input size for the model (e.g. 640 for YOLOv8)
     * @param overlapPercentage the percentage of overlap between tiles (0.0 to 1.0)
     * @param nmsThreshold the threshold for Non-Maximum Suppression (0.0 to 1.0)
     * @throws ModelNotFoundException
     * @throws MalformedModelException
     * @throws IOException
     */
    public DjlObjectDetector(String engine, URI modelUri, Translator<Image, DetectedObjects> translator, int inputSize, double overlapPercentage, double nmsThreshold) 
            throws ModelNotFoundException, MalformedModelException, IOException {
        this.engine = engine;
        this.modelUri = modelUri;
        this.translator = translator;
        this.inputSize = inputSize;
        this.overlapPercentage = Math.min(Math.max(overlapPercentage, 0.0), 1.0);
        this.nmsThreshold = Math.min(Math.max(nmsThreshold, 0.0), 1.0);
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
    
    /**
     * Check if two objects are adjacent (not overlapping)
     * @param obj1 First object
     * @param obj2 Second object
     * @return true if objects are adjacent but not overlapping
     */
    private boolean areObjectsAdjacent(PathObject obj1, PathObject obj2) {
        // Make sure they have the same class
        if (!obj1.getPathClass().equals(obj2.getPathClass())) {
            return false;
        }
        
        // If they overlap, they're not just adjacent
        if (doObjectsOverlap(obj1, obj2)) {
            return false;
        }
        
        qupath.lib.roi.interfaces.ROI roi1 = obj1.getROI();
        qupath.lib.roi.interfaces.ROI roi2 = obj2.getROI();
        
        // Get bounding boxes
        double x1 = roi1.getBoundsX();
        double y1 = roi1.getBoundsY();
        double w1 = roi1.getBoundsWidth();
        double h1 = roi1.getBoundsHeight();
        
        double x2 = roi2.getBoundsX();
        double y2 = roi2.getBoundsY();
        double w2 = roi2.getBoundsWidth();
        double h2 = roi2.getBoundsHeight();
        
        // Maximum distance between objects to be considered adjacent
        double maxDistance = 2.0;
        
        // Check horizontal adjacency with significant vertical overlap
        boolean horizontallyAdjacent = false;
        double horizontalDistance = Math.max(0, Math.max(x1 - (x2 + w2), x2 - (x1 + w1)));
        if (horizontalDistance <= maxDistance) {
            // Calculate vertical overlap
            double yOverlap = Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2);
            double minHeight = Math.min(h1, h2);
            if (yOverlap > 0.8 * minHeight) { // At least 80% vertical overlap
                horizontallyAdjacent = true;
            }
        }
        
        // Check vertical adjacency with significant horizontal overlap
        boolean verticallyAdjacent = false;
        double verticalDistance = Math.max(0, Math.max(y1 - (y2 + h2), y2 - (y1 + h1)));
        if (verticalDistance <= maxDistance) {
            // Calculate horizontal overlap
            double xOverlap = Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2);
            double minWidth = Math.min(w1, w2);
            if (xOverlap > 0.8 * minWidth) { // At least 80% horizontal overlap
                verticallyAdjacent = true;
            }
        }
        
        return horizontallyAdjacent || verticallyAdjacent;
    }
    
    /**
     * Create a new merged PathObject from two adjacent or overlapping objects
     * @param obj1 First object
     * @param obj2 Second object
     * @return A new PathObject with merged bounds
     */
    private PathObject createMergedObject(PathObject obj1, PathObject obj2) {
        qupath.lib.roi.interfaces.ROI roi1 = obj1.getROI();
        qupath.lib.roi.interfaces.ROI roi2 = obj2.getROI();
        
        // Calculate the union bounding box
        double x = Math.min(roi1.getBoundsX(), roi2.getBoundsX());
        double y = Math.min(roi1.getBoundsY(), roi2.getBoundsY());
        double maxX = Math.max(roi1.getBoundsX() + roi1.getBoundsWidth(), 
                               roi2.getBoundsX() + roi2.getBoundsWidth());
        double maxY = Math.max(roi1.getBoundsY() + roi1.getBoundsHeight(), 
                               roi2.getBoundsY() + roi2.getBoundsHeight());
        double width = maxX - x;
        double height = maxY - y;
        
        // Create a new ROI with the merged bounds
        var mergedRoi = ROIs.createRectangleROI(x, y, width, height, roi1.getImagePlane());
        
        // Use the class from the original objects (they should be the same)
        var pathClass = obj1.getPathClass();
        
        // Create a new annotation with the merged ROI
        var mergedObject = PathObjects.createAnnotationObject(mergedRoi, pathClass);
        
        // Calculate weighted average of confidence scores based on area
        double area1 = roi1.getArea();
        double area2 = roi2.getArea();
        double prob1 = obj1.getMeasurementList().get("Class probability");
        double prob2 = obj2.getMeasurementList().get("Class probability");
        double weightedProb = (prob1 * area1 + prob2 * area2) / (area1 + area2);
        
        // Set the confidence score
        mergedObject.getMeasurementList().put("Class probability", weightedProb);
        
        // Add a measurement to indicate this is a merged object (for debugging)
        mergedObject.getMeasurementList().put("Merged", 1.0);
        
        // Log information about the merged objects for debugging
        logger.debug("Merged objects: [x={}, y={}, w={}, h={}] + [x={}, y={}, w={}, h={}] -> [x={}, y={}, w={}, h={}]",
            roi1.getBoundsX(), roi1.getBoundsY(), roi1.getBoundsWidth(), roi1.getBoundsHeight(),
            roi2.getBoundsX(), roi2.getBoundsY(), roi2.getBoundsWidth(), roi2.getBoundsHeight(),
            x, y, width, height);
            
        return mergedObject;
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
            
            // Find all objects to merge with current object
            List<PathObject> overlapping = new ArrayList<>();   // Truly overlapping
            List<PathObject> adjacent = new ArrayList<>();      // Adjacent but not overlapping
            
            for (int j = i + 1; j < sortedDetections.size(); j++) {
                if (!processed[j]) {
                    PathObject other = sortedDetections.get(j);
                    
                    // Separate overlapping from adjacent
                    if (doObjectsOverlap(current, other)) {
                        overlapping.add(other);
                        processed[j] = true;
                    } else if (areObjectsAdjacent(current, other)) {
                        adjacent.add(other);
                        processed[j] = true;
                    }
                }
            }
            
            // Handle truly overlapping objects - keep the one with highest confidence
            if (!overlapping.isEmpty()) {
                PathObject best = current;
                double maxProb = current.getMeasurementList().get("Class probability");
                
                for (PathObject obj : overlapping) {
                    double prob = obj.getMeasurementList().get("Class probability");
                    if (prob > maxProb) {
                        maxProb = prob;
                        best = obj;
                    }
                }
                
                // Start with the best overlapping object
                current = best;
            }
            
            // Handle adjacent objects - merge them with the current object
            PathObject result = current;
            for (PathObject adj : adjacent) {
                result = createMergedObject(result, adj);
            }
            
            merged.add(result);
        }
        
        return merged;
    }

    private List<PathObject> applyNMS(List<DetectedObject> detections, RegionRequest request) {
        final double NMS_THRESHOLD = nmsThreshold;
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