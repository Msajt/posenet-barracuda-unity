using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

[CreateAssetMenu(menuName = "PoseNet Model")]
public class PoseNetModel : ScriptableObject
{
    [Tooltip("The ONNX model asset file to use when perfoming inference")]
    public NNModel modelAsset;

    [Tooltip("The material with the required preprocessing shader")]
    public Material preprocessingMaterial;

    [Tooltip("The index for the heatmap output layer")]
    public int heatmapLayerIndex = 0;
    [Tooltip("The index for the offsets output layer")]
    public int offsetsLayerIndex = 1;
    [Tooltip("The index for the forward displacement layer")]
    public int displacementFWDLayerIndex;
    [Tooltip("The index for the backward displacement layer")]
    public int displacementBWDLayerIndex;
}
