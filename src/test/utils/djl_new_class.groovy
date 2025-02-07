import qupath.ext.djl.DjlObjectDetector
import ai.djl.modality.cv.transform.*
import ai.djl.modality.cv.translator.*
import ai.djl.translate.Pipeline
import static qupath.lib.scripting.QP.*

// Create pipeline for the translator
def pipeline = new Pipeline()
//        .add(new Resize(640, 640))
        .add(new ToTensor())

// Create translator
def translator = new YoloV8Translator.Builder()
        .setPipeline(pipeline)
        .optThreshold(0.5f)
        .build()

// Create file URI for the model
def modelFile = new File('C:/Users/peiya/Downloads/train26/weights/best.torchscript')
def modelUri = modelFile.toURI()

// Create detector with input size 640 and 20% overlap between tiles
def detector = new DjlObjectDetector("PyTorch", modelUri, translator, 640, 0.0)

try {
    // Store existing annotations
    def annotations = getAnnotationObjects()
    
    // Clear only detections, not annotations
    clearDetections()
    
    // Run detection within annotations (if any exist), otherwise run on whole image
    def detections
    if (!annotations.isEmpty()) {
        detections = detector.detect(getCurrentImageData(), annotations)
    } else {
        detections = detector.detect(getCurrentImageData())
    }
    
} catch (Exception e) {
    print "Error: ${e.getMessage()}"
    e.printStackTrace()
} finally {
    detector.close()
}