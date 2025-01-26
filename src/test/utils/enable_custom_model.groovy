import qupath.ext.djl.*
import static qupath.lib.scripting.QP.*
import ai.djl.modality.cv.*
import ai.djl.repository.zoo.*
import ai.djl.translate.*
import ai.djl.modality.cv.translator.*
import ai.djl.modality.cv.transform.*
import java.nio.file.Paths
import ai.djl.modality.cv.output.DetectedObjects
import qupath.lib.objects.PathObject
import qupath.lib.roi.ROIs
import qupath.lib.objects.PathObjects
import qupath.lib.regions.RegionRequest
import qupath.lib.objects.classes.PathClass

// Create file URI directly
def modelFile = new File('C:/Users/peiya/Downloads/train26/weights/best.torchscript')
def modelPath = modelFile.toURI()
println modelPath

def model = null

try {
    def imageData = getCurrentImageData()
    def server = imageData.getServer()
    println "Image dimensions: ${server.getWidth()} x ${server.getHeight()}"
    
    def request = RegionRequest.createInstance(
        server.getPath(),
        1.0,
        0, 0, server.getWidth(), server.getHeight()
    )
    
    Pipeline pipeline = new Pipeline()
            .add(new Resize(640, 640))
            .add(new ToTensor())
    
    translator = new YoloV8Translator.Builder()
            .setPipeline(pipeline)
            .optThreshold(0.5f)
            .build()
            
    // Pass the URI directly to loadModel
    model = DjlTools.loadModel(
        DjlTools.ENGINE_PYTORCH,
        Image.class,
        DetectedObjects.class,
        translator,
        modelPath
    )

    clearAllObjects()
    
    def predictor = model.newPredictor()
    def img = server.readRegion(request)
    def djlImage = ImageFactory.getInstance().fromImage(img)
    def detectedObjects = predictor.predict(djlImage)
    
    // Calculate scaling factor from 640x640 to image size
    double scale = server.getWidth() / 640.0

    // Process each detection
    detectedObjects.items().each { detection ->
        def bbox = detection.getBoundingBox()
        def bounds = bbox.getBounds()
        
        double centerX = bounds.getX() * scale
        double centerY = bounds.getY() * scale
        double width = bounds.getWidth() * scale
        double height = bounds.getHeight() * scale
        
        def roi = ROIs.createRectangleROI(
            centerX,
            centerY,
            width,
            height,
            null
        )
        
        def pathClass = PathClass.fromString(detection.getClassName())
        def annotation = PathObjects.createAnnotationObject(roi, pathClass)
        addObject(annotation)
    }
    
} catch (Exception e) {
    println "Error: ${e.getMessage()}"
    e.printStackTrace()
} finally {
    if (model != null) {
        try {
            model.close()
            println "Model closed successfully"
        } catch (Exception e) {
            println "Error closing model: ${e.getMessage()}"
        }
    }
}