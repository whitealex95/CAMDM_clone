using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Sentis;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace CAMDM
{
    [Serializable]
    public class ConditionParams
    {
        public string[] styles;
    }
    [Serializable]
    public class DiffusionParams
    {
        public int past_points;
        public int future_points;
        public int joint_num;
        public int diffusion_steps;
        public float[] posterior_log_variance_clipped;
        public float[] posterior_mean_coef1;
        public float[] posterior_mean_coef2;
        public string[] joint_names;
        
    }
    
    [Serializable]
    public class DiffusionNetwork
    {
        public DiffusionParams diffusionParams;
        public string[] styles;
        private TensorInt[] m_TimeStepTensor;
        private Model m_Model;
        private Ops m_Ops;
        private ITensorAllocator m_Allocator;
        private IWorker m_Worker;
        private TensorFloat m_EmptyPastMotion;
        private TensorFloat outputTensor;
        private TensorFloat outputTensorCFG;

        private TensorFloat x_0;
        private TensorFloat x_0_uncond;
        private TensorFloat noise;
        private TensorFloat model_mean;
        
        public void CreateSession(ModelAsset modelPath, BackendType device, TextAsset configPath, TextAsset conditionPathh) {
            if(m_Worker != null) 
            {
                Debug.Log("Session is already active.");
            } 
            else {
                m_Model = ModelLoader.Load(modelPath);
                m_Worker = WorkerFactory.CreateWorker(device, m_Model);
                m_Allocator = new TensorCachingAllocator();
                m_Ops = WorkerFactory.CreateOps(device, m_Allocator);
                
                string configJsonContent = configPath.text;
                diffusionParams = JsonUtility.FromJson<DiffusionParams>(configJsonContent);
                
                string conditionJsonContent = conditionPathh.text;
                styles = JsonUtility.FromJson<ConditionParams>(conditionJsonContent).styles;
                
                m_TimeStepTensor = new TensorInt[diffusionParams.diffusion_steps + 1];
                
                for (int i = 0; i <= diffusionParams.diffusion_steps; i++)
                { 
                    m_TimeStepTensor[i] = new TensorInt(new TensorShape(1), new[] { i });
                }
                m_EmptyPastMotion = TensorFloat.Zeros(new TensorShape(1, diffusionParams.joint_num+1, 6, diffusionParams.past_points));
            }
        }
        
        public TensorFloat Inference(TensorFloat pastMotion, TensorFloat trajPose, TensorFloat trajTrans, TensorFloat styleFearture, float guidanceScale)
        {
            TensorFloat x_t = m_Ops.RandomNormal(new TensorShape(1, diffusionParams.joint_num+1, 6, diffusionParams.future_points), 0f,1f, null);
            
            Dictionary<string, Tensor> inputTensor = new Dictionary<string, Tensor>()
            {
                { "input_x", x_t },
                { "time_steps", null},
                { "past_motion", null },
                { "traj_pose", trajPose },
                { "traj_trans", trajTrans },
                { "style_idx", styleFearture },
            };
            
            for (int i = diffusionParams.diffusion_steps-1; i >= 0; i--)
            {
                inputTensor["input_x"] = x_t;
                inputTensor["time_steps"] = m_TimeStepTensor[i];
                inputTensor["past_motion"] = pastMotion;
                m_Worker.Execute(inputTensor);    
                outputTensor = m_Worker.PeekOutput("output") as TensorFloat;
                // x_0 takes ownership of a copy of the output; release the peeked tensor
                x_0?.Dispose();
                x_0 = m_Ops.Mul(1f, outputTensor);
                outputTensor?.Dispose();

                if (guidanceScale != 1f)
                {
                    inputTensor["input_x"] = x_t;
                    inputTensor["time_steps"] = m_TimeStepTensor[i];
                    inputTensor["past_motion"] = m_EmptyPastMotion;
                    m_Worker.Execute(inputTensor); 
                    outputTensorCFG = m_Worker.PeekOutput("output") as TensorFloat;
                    x_0_uncond?.Dispose();
                    x_0_uncond = m_Ops.Mul(1f, outputTensorCFG);
                    outputTensorCFG?.Dispose();
                    // Break down ops to dispose intermediates
                    using var diff = m_Ops.Sub(x_0, x_0_uncond);
                    using var scaled = m_Ops.Mul(guidanceScale, diff);
                    using var baseUncond = m_Ops.Mul(1f, x_0_uncond);
                    var combined = m_Ops.Add(baseUncond, scaled);
                    x_0.Dispose();
                    x_0 = combined;
                    x_0_uncond.Dispose();
                }

                float model_log_variance = diffusionParams.posterior_log_variance_clipped[i];
                float mean_coef1 = diffusionParams.posterior_mean_coef1[i];
                float mean_coef2 = diffusionParams.posterior_mean_coef2[i];
                model_mean?.Dispose();
                using var mm1 = m_Ops.Mul(mean_coef1, x_0);
                using var mm2 = m_Ops.Mul(mean_coef2, x_t);
                model_mean = m_Ops.Add(mm1, mm2);

                if (i > 0)
                {
                    noise?.Dispose();
                    noise = m_Ops.RandomNormal(new TensorShape(1, diffusionParams.joint_num+1, 6, diffusionParams.future_points), 0f, 1f, null);
                    using var scaledNoise = m_Ops.Mul(Mathf.Exp(0.5f * model_log_variance), noise);
                    var new_xt = m_Ops.Add(model_mean, scaledNoise); // x_{t-1}
                    // replace x_t
                    x_t.Dispose();
                    x_t = new_xt;
                    // model_mean and noise no longer needed in this branch
                    model_mean.Dispose();
                    noise.Dispose();
                    model_mean = null;
                    noise = null;
                }
                else
                {
                    // replace x_t with model_mean; do not dispose model_mean yet as it's now referenced by x_t
                    x_t.Dispose();
                    x_t = model_mean;
                    model_mean = null;
                }
            }
            pastMotion.MakeReadable();
            trajPose.MakeReadable();
            trajTrans.MakeReadable();
            x_t.MakeReadable();
            styleFearture.MakeReadable();
            return x_t;
        }
        
        public void Dispose() {
            if(m_Worker != null) {
                m_Worker.Dispose();
                m_Worker = null;
            }
            if(m_Allocator != null) {
                m_Allocator.Dispose();
                m_Allocator = null;
            }
            if(m_Ops != null) {
                m_Ops.Dispose();
                m_Ops = null;
            }
            
            if(m_TimeStepTensor != null) {
                for(int i=0; i<m_TimeStepTensor.Length; i++) {
                    if(m_TimeStepTensor[i] != null) {
                        m_TimeStepTensor[i].Dispose();
                        m_TimeStepTensor[i] = null;
                    }
                }
            }
            
            m_EmptyPastMotion?.Dispose();
            outputTensor?.Dispose();
            outputTensorCFG?.Dispose();
            x_0?.Dispose();
            x_0_uncond?.Dispose();
            noise?.Dispose();
            model_mean?.Dispose();
        }
        
        public void OnDestroy() {
            Dispose();
        }
        
        #if UNITY_EDITOR
        public bool Inspector() {
            EditorGUI.BeginChangeCheck();
            Utility.SetGUIColor(UltiDraw.White);
            using(new EditorGUILayout.VerticalScope ("Box")) {
                Utility.ResetGUIColor();
            }
            return EditorGUI.EndChangeCheck();
        }
        #endif
    }
}

