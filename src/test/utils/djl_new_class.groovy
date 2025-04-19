import qupath.ext.djl.DjlObjectDetector
import ai.djl.modality.cv.transform.*
import ai.djl.modality.cv.translator.*
import ai.djl.translate.Pipeline
import qupath.lib.roi.RoiTools
import java.awt.geom.Area
import static qupath.lib.scripting.QP.*

// Create pipeline for the translator
def pipeline = new Pipeline().add(new ToTensor())

// Create translator
def translator = new YoloV8Translator.Builder()
        .setPipeline(pipeline)
        .optThreshold(0.01f)
        .build()

// Load model
def modelFile = new File("C:/Users/peiya/Desktop/train16/weights/best.torchscript")
def modelUri = modelFile.toURI()

// Create detector
def detector = new DjlObjectDetector("PyTorch", modelUri, translator, 640, 0.0, 0.45, 0.01)
detector.setClassThreshold("TA", 0.06)
detector.setClassThreshold("CB", 0.06)

try {
    // Clear previous detections
    clearDetections()

    // Get selected annotations
    def selectedAnnotations = getSelectedObjects().findAll { it.isAnnotation() }

    if (selectedAnnotations.isEmpty()) {
        print "No annotations selected! Please select one or more annotations."
        return
    }

    // Run detection within selected annotations
    def detections = detector.detect(getCurrentImageData(), selectedAnnotations)

    if (detections.isPresent()) {
        print "Detection completed with ${detections.get().size()} objects found."
    } else {
        print "Detection was interrupted."
    }

    def combinedArea = new Area()
    for (ann in selectedAnnotations) {
        def shape = RoiTools.getShape(ann.getROI())
        combinedArea.add(new Area(shape))
    }

    // Get all annotations excluding the selected ones
    def otherAnnotations = getAnnotationObjects().findAll { !selectedAnnotations.contains(it) }

    // Keep annotations that intersect the selected area
    def annotationsToKeep = otherAnnotations.findAll { ann ->
        def annArea = new Area(RoiTools.getShape(ann.getROI()))
        annArea.intersect(new Area(combinedArea)) // Make a copy for safe intersection
        return !annArea.isEmpty()
    }

    // Remove annotations that are fully outside
    def annotationsToRemove = otherAnnotations - annotationsToKeep
    removeObjects(annotationsToRemove, true)

    print "Removed ${annotationsToRemove.size()} annotations outside the selected region(s)."
} finally {
    detector.close()
}
