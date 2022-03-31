using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class PoseEstimator : MonoBehaviour
{
    //TODO ===========================================
    //TODO ===== P U B L I C   V A R I A B L E S =====
    //TODO ===========================================
    public enum EstimationType{
        MultiPose,
        SinglePose
    }

    [Tooltip("The requested webcam dimensions")]
    public Vector2Int webcamDims = new Vector2Int(1280, 720);

    [Tooltip("The requested webcam frame rate")]
    public int webcamFPS = 60;

    [Tooltip("The screen for viewing the input source")]
    public Transform videoScreen;

    [Tooltip("Stores the models available for inference")]
    public PoseNetModel[] models;

    [Tooltip("The dimensions of the image being fed to the model")]
    public Vector2Int imageDims = new Vector2Int(256, 256);

    [Tooltip("The backend to use when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Tooltip("The type of pose estimation to be performed")]
    public EstimationType estimationType = EstimationType.SinglePose;

    [Tooltip("The maximum number of posees to estimate")]
    [Range(1, 20)]
    public int maxPoses = 20;

    [Tooltip("The score threshold for multipose estimation")]
    [Range(0, 1.0f)]
    public float scoreThreshold = 0.25f;

    [Tooltip("Non-maximum suppression distance")]
    public int nmsRadius = 100;

    [Tooltip("The size of the pose skeleton key points")]
    public float pointScale = 10f;

    [Tooltip("The width of the pose skeleton lines")]
    public float lineWidth = 5f;

    [Tooltip("The minimum confidence level required to display a key point")]
    [Range(0, 100)]
    public int minConfidence = 70;

    //TODO =============================================
    //TODO ===== P R I V A T E   V A R I A B L E S =====
    //TODO =============================================
    //! Live video input from a webcam
    private WebCamTexture webcamTexture;

    //! The source video texture
    private RenderTexture videoTexture;

    //! Target dimensions for model input
    private Vector2Int targetDims;

    //! Used to scale the input image dimensions while maintaining aspect ratio
    private float aspectRatioScale;

    //! The texture used to create the input tensor
    private RenderTexture rTex;

    //! Stores the input data for the model
    private Tensor input;

    //! The interface used to execute the neural network
    private IWorker engine;

    //! The name for the heatmap layer in the model asset
    private string heatmapLayer;

    //! The name for the offsets layer in the model asset
    private string offsetsLayer;

    //! The name for the forwards displacement layer in the model asset
    private string displacementFWDLayer;

    //! The name for the backwards displacement layer in the model asset
    private string displacementBWDLayer;

    //! The name for the Sigmoid layer that returns the heatmap predictions
    private string predictionLayer = "heatmap_predictions";

    //! Stores the current estimated 2D keypoint locations in videoTexture
    private Utils.Keypoint[][] poses;

    //! Array of pose skeletons
    private PoseSkeleton[] skeletons;

    //! Stores the PoseNetModel currently in use
    private PoseNetModel currentModel;

    //TODO ===================================
    //TODO ===== INITIALIZE VIDEO SCREEN =====
    //TODO ===================================

    //! Prepares the videoScreen GameObject to display the chosen video source
    private void InitializeVideoScreen(int width, int height, bool mirrorScreen){
        //? Release temporary RenderTexture
        RenderTexture.ReleaseTemporary(videoTexture);
        //? Create a new videoTexture using the current video dimensions
        videoTexture = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGBHalf);

        if (mirrorScreen)
        {
            //* Flip the VideoScreen around the Y-Axis
            videoScreen.rotation = Quaternion.Euler(0, 180, 0);
            //* Invert the scale value for the Z-Axis
            videoScreen.localScale = new Vector3(videoScreen.localScale.x, videoScreen.localScale.y, -1f);
        }

        //? Apply the new videoTexture to the VideoScreen Gameobject
        videoScreen.gameObject.GetComponent<MeshRenderer>().material.shader = Shader.Find("Unlit/Texture");
        videoScreen.gameObject.GetComponent<MeshRenderer>().material.SetTexture("_MainTex", videoTexture);
        //? Adjust the VideoScreen dimensions for the new videoTexture
        videoScreen.localScale = new Vector3(width, height, videoScreen.localScale.z);
        //? Adjust the VideoScreen position for the new videoTexture
        videoScreen.position = new Vector3(width / 2, height / 2, 1);
    }

    //TODO =============================
    //TODO ===== INITIALIZE CAMERA =====
    //TODO =============================

    //! Resizes and positions the in-game Camera to accomodate the video dimensions
    private void InitializeCamera(){
        //? Get a reference to the Main Camera GameObject
        GameObject mainCamera = GameObject.Find("Main Camera");
        //? Adjust the camera position to account for updates to the VideoScreen
        mainCamera.transform.position = new Vector3(videoTexture.width / 2, videoTexture.height / 2, -10f);
        //? Render objects with no perspective (i.e. 2D)
        mainCamera.GetComponent<Camera>().orthographic = true;
        //? Adjust the camera size to account for updates to the VideoScreen
        mainCamera.GetComponent<Camera>().orthographicSize = videoTexture.height / 2;
    }

    //TODO ================================
    //TODO ===== INITIALIZE BARRACUDA =====
    //TODO ================================

    //! Updates the output layer names based on the selected model architecture
    //! and initializes the Barracuda inference engine with the selected model.
    private void InitializeBarracuda(){
        //? The compiled model used for performing inference
        Model m_RunTimeModel;

        //? Compile the model asset into an object oriented representation
        m_RunTimeModel = ModelLoader.Load(currentModel.modelAsset);

        //? Get output layer names
        heatmapLayer = m_RunTimeModel.outputs[currentModel.heatmapLayerIndex];
        offsetsLayer = m_RunTimeModel.outputs[currentModel.offsetsLayerIndex];
        displacementFWDLayer = m_RunTimeModel.outputs[currentModel.displacementFWDLayerIndex];
        displacementBWDLayer = m_RunTimeModel.outputs[currentModel.displacementBWDLayerIndex];

        //? Create a model builder to modify the m_RunTimeModel
        ModelBuilder modelBuilder = new ModelBuilder(m_RunTimeModel);

        //? Add a new Sigmoid layer that takes the output of the heatmap layer
        modelBuilder.Sigmoid(predictionLayer, heatmapLayer);

        //? Validate if backend is supported, otherwise use fallback type.
        workerType = WorkerFactory.ValidateType(workerType);

        //? Create a worker that will execute the model with the selected backend
        engine = WorkerFactory.CreateWorker(workerType, modelBuilder.model);
    }

    //TODO ================================
    //TODO ===== INITIALIZE SKELETONS =====
    //TODO ================================

    //! Initialize pose skeletons
    private void InitializeSkeletons(){
        //? Initialize the list of pose skeletons
        if (estimationType == EstimationType.SinglePose) maxPoses = 1;
        skeletons = new PoseSkeleton[maxPoses];

        //? Populate the list of pose skeletons
        for (int i = 0; i < maxPoses; i++) skeletons[i] = new PoseSkeleton(pointScale, lineWidth);    
    }

    //TODO =================================
    //TODO ===== INITIALIZE INPUT DIMS =====
    //TODO =================================

    //! Initialize the input dimensions for the model
    private void InitializeInputDims(){
        //? Prevent the input dimensions from going too low for the model
        imageDims.x = Mathf.Max(imageDims.x, 64);
        imageDims.y = Mathf.Max(imageDims.y, 64);

        //? Update the input dimensions while maintaining the source aspect ratio
        if (imageDims.y != targetDims.y)
        {
            aspectRatioScale = (float)videoTexture.width / videoTexture.height;
            targetDims.x = (int)(imageDims.y * aspectRatioScale);
            imageDims.x = targetDims.x;
            targetDims.y = imageDims.y;
        }

        if (imageDims.x != targetDims.x)
        {
            aspectRatioScale = (float)videoTexture.height / videoTexture.width;
            targetDims.y = (int)(imageDims.x * aspectRatioScale);
            imageDims.y = targetDims.y;
            targetDims.x = imageDims.x;
        }

        RenderTexture.ReleaseTemporary(rTex);
        //? Assign a temporary RenderTexture with the new dimensions
        rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, RenderTextureFormat.ARGBHalf);
    }

    //TODO =====================================
    //TODO ===== INITIALIZE POSE ESTIMATOR =====
    //TODO =====================================

    //! Perform initialization steps
    private void InitializePoseEstimator(){
        //? Initialize the Barracuda inference engine based on the selected model
        InitializeBarracuda();

        //? Initialize pose skeletons
        InitializeSkeletons();

        //? Initialize the videoScreen
        InitializeVideoScreen(webcamTexture.width, webcamTexture.height, true);

        //? Adjust the camera based on the source video dimensions
        InitializeCamera();

        //? Initialize input dimensions
        InitializeInputDims();    
    }

    //TODO =================
    //TODO ===== START =====
    //TODO =================

    void Start(){
        //? Create a new WebCamTexture
        webcamTexture = new WebCamTexture(webcamDims.x, webcamDims.y, webcamFPS);
        //? Start the Camera
        webcamTexture.Play();

        //? Default to the first PoseNetModel in the list
        currentModel = models[0];
    }

    //TODO =========================
    //TODO ===== PROCESS IMAGE =====
    //TODO =========================

    //! Calls the appropriate preprocessing function to prepare
    //! the input for the selected model
    private void ProcessImage(RenderTexture image){
        //? Define a temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        RenderTexture.active = result;

        //? Apply preprocessing steps
        Graphics.Blit(image, result, currentModel.preprocessingMaterial);

        //? Create a new Tensor
        input = new Tensor(result, channels: 3);
        RenderTexture.ReleaseTemporary(result);
    }

    //TODO ========================
    //TODO ==== PROCESS OUTPUT ====
    //TODO ========================

    //! Obtains the model output and either decodes single or multiple poses
    private void ProcessOutput(IWorker engine){
        //? Get the model output
        Tensor heatmaps = engine.PeekOutput(predictionLayer);
        Tensor offsets = engine.PeekOutput(offsetsLayer);
        Tensor displacementFWD = engine.PeekOutput(displacementFWDLayer);
        Tensor displacementBWD = engine.PeekOutput(displacementBWDLayer);

        //? Calculate the stride used to scale down the inputImage
        int stride = (imageDims.y - 1) / (heatmaps.shape.height - 1);
        stride -= (stride % 8);

        if (estimationType == EstimationType.SinglePose)
        {
            //* Initialize the array of Keypoint arrays
            poses = new Utils.Keypoint[1][];

            //* Determine the key point locations
            poses[0] = Utils.DecodeSinglePose(heatmaps, offsets, stride);
        }
        else
        {
            //* Determine the key point locations
            poses = Utils.DecodeMultiplePoses(
                heatmaps, offsets,
                displacementFWD, displacementBWD,
                stride: stride, maxPoseDetections: maxPoses,
                scoreThreshold: scoreThreshold,
                nmsRadius: nmsRadius);
        }

        //? Release the resources allocated for the output Tensors
        heatmaps.Dispose();
        offsets.Dispose();
        displacementFWD.Dispose();
        displacementBWD.Dispose();
    }

    //TODO ==================
    //TODO ===== UPDATE =====
    //TODO ==================
    void Update(){
        //? Skip the rest of the method if the webcam is not initialized
        if (webcamTexture.width <= 16) return;

        //? Only perform initialization steps if the videoTexture has not been initialized
        if (!videoTexture) InitializePoseEstimator();

        //? Copy webcamTexture to videoTexture
        Graphics.Blit(webcamTexture, videoTexture);

        //? Copy the videoTexture data to rTex
        Graphics.Blit(videoTexture, rTex);

        //? Prepare the input image to be fed to the selected model
        ProcessImage(rTex);

        //? Execute neural network with the provided input
        engine.Execute(input);
        //? Release GPU resources allocated for the Tensor
        input.Dispose();

        //? Decode the keypoint coordinates from the model output
        ProcessOutput(engine);

        //? Reinitialize pose skeletons
        if (maxPoses != skeletons.Length){
            foreach (PoseSkeleton skeleton in skeletons){
                skeleton.Cleanup();
            }

            //* Initialize pose skeletons
            InitializeSkeletons();
        }

        //? The smallest dimension of the videoTexture
        int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);

        //? The value used to scale the key point locations up to the source resolution
        float scale = (float)minDimension / Mathf.Min(imageDims.x, imageDims.y);

        //? Update the pose skeletons
        for (int i = 0; i < skeletons.Length; i++){
            if (i <= poses.Length - 1){
                skeletons[i].ToggleSkeleton(true);

                //* Update the positions for the key point GameObjects
                skeletons[i].UpdateKeyPointPositions(poses[i], scale, videoTexture, true, minConfidence);
                skeletons[i].UpdateLines();
            } else{
                skeletons[i].ToggleSkeleton(false);
            }
        }

        Resources.UnloadUnusedAssets();
    }

    //TODO ======================
    //TODO ===== ON DISABLE =====
    //TODO ======================
    
    //! OnDisable is called when the MonoBehavior becomes disabled or inactive
    private void OnDisable(){
        //? Release the resources allocated for the inference engine
        engine.Dispose();
    }
}

