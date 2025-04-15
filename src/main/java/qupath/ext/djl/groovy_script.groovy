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
        .optThreshold(0.5f)
        .build()

// Create file URI for the model
def modelFile = new File('C:/Users/peiya/Downloads/train26/weights/best.torchscript')
def modelUri = modelFile.toURI()

// Create detector with input size 640
def detector = new DjlObjectDetector("PyTorch", modelUri, translator, 640, 0.0)

try {
    // Get the selected object
    def selected = getSelectedObject()
    if (selected == null) {
        print("Please select an annotation first!")
        return
    }

    // Clear child objects of the selected annotation
    clearDetections()
    
    // Run detection only on the selected annotation
    def detections = detector.detect(getCurrentImageData(), [selected])
} finally {
    detector.close()
}