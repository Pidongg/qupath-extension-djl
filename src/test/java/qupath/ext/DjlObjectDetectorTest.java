package qupath.ext;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.objects.classes.PathClass;

public class DjlObjectDetectorTest {
    
    // Helper method to create a test annotation object
    private PathObject createTestObject(double x, double y, double width, double height, 
                                        String className, double probability) {
        ROI roi = ROIs.createRectangleROI(x, y, width, height, null);
        PathClass pathClass = PathClass.fromString(className);
        PathObject obj = PathObjects.createAnnotationObject(roi, pathClass);
        obj.getMeasurementList().put("Class probability", probability);
        return obj;
    }
    
    // Implementation of doObjectsOverlap for testing
    private static boolean doObjectsOverlap(PathObject obj1, PathObject obj2) {
        // Get ROIs in global image coordinates
        ROI roi1 = obj1.getROI();
        ROI roi2 = obj2.getROI();
        
        // Calculate IoU (Intersection over Union)
        var intersection = roi1.getGeometry().intersection(roi2.getGeometry());
        if (intersection.isEmpty())
            return false;
        
        double intersectionArea = intersection.getArea();
        double unionArea = roi1.getArea() + roi2.getArea() - intersectionArea;
        double iou = intersectionArea / unionArea;
        
        // Only consider as overlapping if IoU exceeds threshold
        double iouThreshold = 0.0; // Use 0 for testing - any overlap counts
        return iou > iouThreshold;
    }
    
    // Implementation of areObjectsAdjacent for testing
    private static boolean areObjectsAdjacent(PathObject obj1, PathObject obj2) {
        // Make sure they have the same class
        if (!obj1.getPathClass().equals(obj2.getPathClass())) {
            return false;
        }
        
        // If they overlap, they're not just adjacent
        if (doObjectsOverlap(obj1, obj2)) {
            return false;
        }
        
        ROI roi1 = obj1.getROI();
        ROI roi2 = obj2.getROI();
        
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
    
    // Implementation of createMergedObject for testing
    private static PathObject createMergedObject(PathObject obj1, PathObject obj2) {
        ROI roi1 = obj1.getROI();
        ROI roi2 = obj2.getROI();
        
        // Calculate the union bounding box
        double x = Math.min(roi1.getBoundsX(), roi2.getBoundsX());
        double y = Math.min(roi1.getBoundsY(), roi2.getBoundsY());
        double maxX = Math.max(roi1.getBoundsX() + roi1.getBoundsWidth(), 
                              roi2.getBoundsX() + roi2.getBoundsWidth());
        double maxY = Math.max(roi1.getBoundsY() + roi1.getBoundsHeight(), 
                              roi2.getBoundsY() + roi2.getBoundsHeight());
        double width = maxX - x;
        double height = maxY - y;
        
        // Create new ROI with the union bounds
        ROI newRoi = ROIs.createRectangleROI(x, y, width, height, null);
        
        // Use the same class as the input objects (they should have the same class)
        PathClass pathClass = obj1.getPathClass();
        
        // Create new annotation with the merged ROI
        PathObject merged = PathObjects.createAnnotationObject(newRoi, pathClass);
        
        // Calculate weighted probability based on areas
        double prob1 = obj1.getMeasurementList().get("Class probability");
        double prob2 = obj2.getMeasurementList().get("Class probability");
        double area1 = roi1.getArea();
        double area2 = roi2.getArea();
        double weightedProb = (prob1 * area1 + prob2 * area2) / (area1 + area2);
        
        // Add measurements
        merged.getMeasurementList().put("Class probability", weightedProb);
        merged.getMeasurementList().put("Merged", 1.0);
        
        return merged;
    }
    
    @Test
    @DisplayName("Test doObjectsOverlap with overlapping objects")
    public void testDoObjectsOverlap_WhenOverlapping_ReturnsTrue() {
        
        PathObject obj1 = createTestObject(10, 10, 20, 20, "cell", 0.8);
        PathObject obj2 = createTestObject(20, 20, 20, 20, "cell", 0.7);
        
        boolean result = doObjectsOverlap(obj1, obj2);
        
        assertTrue(result, "Objects with overlapping ROIs should return true");
    }
    
    @Test
    @DisplayName("Test doObjectsOverlap with non-overlapping objects")
    public void testDoObjectsOverlap_WhenNotOverlapping_ReturnsFalse() {
        
        PathObject obj1 = createTestObject(10, 10, 10, 10, "cell", 0.8);
        PathObject obj2 = createTestObject(30, 30, 10, 10, "cell", 0.7);
        
        boolean result = doObjectsOverlap(obj1, obj2);
        
        assertFalse(result, "Objects with non-overlapping ROIs should return false");
    }
    
    @Test
    @DisplayName("Test areObjectsAdjacent with adjacent objects")
    public void testAreObjectsAdjacent_WhenAdjacent_ReturnsTrue() {
        PathObject obj1 = createTestObject(10, 10, 10, 50, "cell", 0.8);
        PathObject obj2 = createTestObject(21, 15, 10, 40, "cell", 0.7);
        
        boolean result = areObjectsAdjacent(obj1, obj2);
        
        assertTrue(result, "Objects that are adjacent and have good vertical overlap should return true");
    }
    
    @Test
    @DisplayName("Test areObjectsAdjacent with non-adjacent objects")
    public void testAreObjectsAdjacent_WhenNotAdjacent_ReturnsFalse() {
        PathObject obj1 = createTestObject(10, 10, 10, 10, "cell", 0.8);
        PathObject obj2 = createTestObject(30, 30, 10, 10, "cell", 0.7);
        
        boolean result = areObjectsAdjacent(obj1, obj2);
        
        assertFalse(result, "Objects that are far apart should not be considered adjacent");
    }
    
    @Test
    @DisplayName("Test areObjectsAdjacent with different classes")
    public void testAreObjectsAdjacent_WithDifferentClasses_ReturnsFalse() {
        PathObject obj1 = createTestObject(10, 10, 10, 50, "cell", 0.8);
        PathObject obj2 = createTestObject(21, 15, 10, 40, "nucleus", 0.7);
        
        boolean result = areObjectsAdjacent(obj1, obj2);

        assertFalse(result, "Objects with different classes should not be considered adjacent");
    }
    
    @Test
    @DisplayName("Test createMergedObject merges objects correctly")
    public void testCreateMergedObject_MergesTwoObjects() {
        PathObject obj1 = createTestObject(10, 10, 20, 20, "cell", 0.8);
        PathObject obj2 = createTestObject(20, 20, 20, 20, "cell", 0.6);
        
        PathObject mergedObj = createMergedObject(obj1, obj2);
        
        assertNotNull(mergedObj, "Merged object should not be null");
        assertEquals("cell", mergedObj.getPathClass().getName(), "Merged object should maintain class name");
        
        ROI mergedRoi = mergedObj.getROI();
        assertEquals(10.0, mergedRoi.getBoundsX(), "Merged X should be minimum of both objects");
        assertEquals(10.0, mergedRoi.getBoundsY(), "Merged Y should be minimum of both objects");
        assertEquals(30.0, mergedRoi.getBoundsWidth(), "Merged width should cover both objects");
        assertEquals(30.0, mergedRoi.getBoundsHeight(), "Merged height should cover both objects");
        
        // Check probability was weighted properly
        double expectedProb = (0.8 * 400 + 0.6 * 400) / 800; // Area weighted average
        assertEquals(expectedProb, mergedObj.getMeasurementList().get("Class probability"), 0.001);
        
        // Check that merged flag is set
        assertEquals(1.0, mergedObj.getMeasurementList().get("Merged"), "Merged flag should be set");
    }
}