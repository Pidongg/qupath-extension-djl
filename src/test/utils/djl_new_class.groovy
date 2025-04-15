import qupath.ext.djl.DjlObjectDetector
import ai.djl.modality.cv.transform.*
import ai.djl.modality.cv.translator.*
import ai.djl.translate.Pipeline
import static qupath.lib.scripting.QP.*

// Create pipeline for the translator
def pipeline = new Pipeline()
        .add(new ToTensor())

// Create translator
def translator = new YoloV8Translator.Builder()
        .setPipeline(pipeline)
        .optThreshold(0.24f)
        .build()

// Create file URI for the model
def modelFile = new File("C:/Users/peiya/Desktop/train16/weights/best.torchscript")
def modelUri = modelFile.toURI()

// Create detector with input size 640, overlap percentage between adjacent tiles 0, iou threshold 0.45
def detector = new DjlObjectDetector("PyTorch", modelUri, translator, 640, 0.0, 0.45)

try {
    // Clear existing detections
    clearDetections()
    
    // Get selected annotations
    def selectedAnnotations = getSelectedObjects()
    
    // If no annotations selected, show warning
    if (selectedAnnotations.isEmpty()) {
        print("No annotations selected! Please select one or more annotations.")
        return
    }
    
    // Run detection only on selected annotations
    def detections = detector.detect(getCurrentImageData(), selectedAnnotations)
    
    if (detections.isPresent()) {
        print("Detection completed with ${detections.get().size()} objects found")
    } else {
        print("Detection was interrupted")
    }
    
} finally {
    detector.close()
}