using UnityEngine;
using System;
using System.Collections.Generic;
using TPSBR.Core;
using System.Collections.Concurrent;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace TPSBR
{
    [System.Serializable]
    public class ILState
    {
        // Agent state variables
        public Vector3 agentPosition;
        public Vector3 agentRotation;
        public Vector3 agentForward;
        public float health;
        public float weapon;
        
        // Target state variables
        public Vector3 targetPosition;
        public Vector3 targetRotation;
        public Vector3 targetForward;
        
        // Relationship variables
        public Vector3 directionToTarget;
        public Vector3 cross;
        public float distance;
        public float dotProduct;
        public bool islos;
    }

    [System.Serializable]
    public class ILStatePayload
    {
        public ILState state;
    }

    [System.Serializable]
    public class ILAPIResponse
    {
        public float[] predictions;
    }

    public class ILInput : MonoBehaviour
    {
        [Header("API Configuration")]
            [SerializeField] private string apiUrl = "http://localhost:8000/predict/unity";
        [SerializeField] private bool enableAPIInput = true;
        [SerializeField] private float apiCallInterval = 0.05f; // Call API every 0.1 seconds
        
        [Header("Debug")]
        [SerializeField] private bool enableDebugLogs = true;
        
        // Core components
        private Agent agent;
        private GameObject currentTargetAgentObject;
        private AIController aiController;
        
        // API communication
        private HttpClient httpClient = new HttpClient();
            private float[] cachedPredictions = new float[2] { 0f, 0f };
        
        // Timing
        private float lastAPICallTime = 0f;
        private bool isAPICallInProgress = false;
        
        // Output values for AgentInput
        public Vector2 AIMovementDirection { get; private set; } = Vector2.zero;
        public Vector2 AILookRotationDelta { get; private set; } = Vector2.zero;
        public bool AIShouldAttack { get; private set; } = false;
        
        // World-space movement target derived from server deltas
        private bool _hasWorldTarget = false;
        private Vector3 _targetWorldPosition = Vector3.zero;
        [SerializeField] private float targetReachThreshold = 0.25f; // world units
        [SerializeField] private bool debugLogMovement = true;
        [SerializeField] private float arriveBrakingDistance = 3.0f; // slow down when closer than this
        [SerializeField] private bool snapAtGoal = false; // snap to exact target when extremely close
        [SerializeField] private float precisionSnapThreshold = 0.01f; // snap threshold (xz)
        
        // Running average for movement inputs
        [SerializeField] private int movementAverageWindow = 20; // number of frames to average
        private readonly System.Collections.Generic.Queue<Vector2> _recentMovementInputs = new System.Collections.Generic.Queue<Vector2>();
        private Vector2 _recentMovementSum = Vector2.zero;
        
        void Start()
        {
            InitializeComponents();
        }
        
        private Vector2 GetAveragedMovement(Vector2 rawInput)
        {
            _recentMovementInputs.Enqueue(rawInput);
            _recentMovementSum += rawInput;

            while (_recentMovementInputs.Count > movementAverageWindow)
            {
                Vector2 oldest = _recentMovementInputs.Dequeue();
                _recentMovementSum -= oldest;
            }

            int count = _recentMovementInputs.Count;
            if (count <= 0)
            {
                return rawInput;
            }

            return _recentMovementSum / count;
        }
        
        void Update()
        {
            if (!enableAPIInput || agent == null)
                return;
                
            // If we have a world-space target, drive local inputs to move toward it each frame
            if (_hasWorldTarget)
            {
                Vector3 currentPos = agent.transform.position;
                Vector3 toTarget = _targetWorldPosition - currentPos;
                // toTarget.y = 0f; // ignore vertical offset
                float distance = toTarget.magnitude;

                // Optional precise snap when extremely close
                // if (snapAtGoal && distance <= precisionSnapThreshold)
                // {
                //     var p = agent.transform.position;
                //     agent.transform.position = new Vector3(_targetWorldPosition.x, p.y, _targetWorldPosition.z);
                //     _hasWorldTarget = false;
                //     AIMovementDirection = Vector2.zero;
                //     if (debugLogMovement) GlipDebug.Log($"[ILInput] Snapped to target {_targetWorldPosition}");
                // }
                // else if (distance <= targetReachThreshold)
                // {
                //     _hasWorldTarget = false;
                //     AIMovementDirection = Vector2.zero;
                //     if (debugLogMovement) GlipDebug.Log($"[ILInput] Reached target {_targetWorldPosition}");
                // }
                // else
                // {
                    // Vector3 worldDir = toTarget / Mathf.Max(distance, 0.0001f);
                    Vector3 worldDir = toTarget ;
                    // Convert world direction to local input axes (x: strafe, z: forward)
                    Vector3 localDir = agent.transform.InverseTransformDirection(worldDir);
                    // Arrival braking: scale input by proximity so we slow down as we approach
                    float scale = Mathf.Clamp01(distance / Mathf.Max(arriveBrakingDistance, 0.0001f));
                    Vector2 inputDir = new Vector2(localDir.x, localDir.z);
                    // if (inputDir.sqrMagnitude > 1f) inputDir.Normalize();
                    // inputDir *= scale;
                    // Clamp to valid input range
                    // inputDir.x = Mathf.Clamp(inputDir.x, -1f, 1f);
                    // inputDir.y = Mathf.Clamp(inputDir.y, -1f, 1f);
                    AIMovementDirection = GetAveragedMovement(inputDir * 5);
                    if (debugLogMovement) GlipDebug.Log($"[ILInput] Moving toward {_targetWorldPosition}, distance {distance:F2}, input {AIMovementDirection}");
                // }
            }

            // Check if it's time to make an API call
            if (Time.time - lastAPICallTime >= apiCallInterval && !isAPICallInProgress)
            {
                    var state = CreateCurrentState();
                    _ = SendStateToAPI(state);
            }
        }
        
        private void InitializeComponents()
        {
            agent = GetComponent<Agent>();
            if (agent == null)
            {
                LogDebug("Agent component not found on this GameObject.", true);
                enabled = false;
                return;
            }
            
            aiController = GetComponent<AIController>();
            if (aiController == null)
            {
                LogDebug("AIController component not found on this GameObject.", true);
            }

            // Only run ILInput for the local player (has input authority) and skip AI bots
            if (agent.Object == null || !agent.Object.HasInputAuthority || agent.isAIBot)
            {
                LogDebug("ILInput disabled because the agent either lacks input authority or is a bot.");
                enabled = false;
                return;
            }
 
            LogDebug("ILInput initialized successfully");
        }
        
        private ILState CreateCurrentState()
        {
            if (agent == null)
            {
                return new ILState
                {
                    agentPosition = Vector3.zero,
                    agentRotation = Vector3.zero,
                    agentForward = Vector3.zero,
                    health = 0f,
                    weapon = 0f,
                    targetPosition = Vector3.zero,
                    targetRotation = Vector3.zero,
                    targetForward = Vector3.zero,
                    directionToTarget = Vector3.zero,
                    cross = Vector3.zero,
                    distance = 0f,
                    dotProduct = 0f,
                    islos = false
                };
            }

            try
            {
                currentTargetAgentObject = agent.ReturnTargetAgent(agent, 100000.0f);

                // Agent state (raw, server will normalize)
                Vector3 agentPosition = agent.transform.position;
                Vector3 agentRotation = agent.transform.eulerAngles;
                Vector3 agentForward = agent.transform.forward;
                float health = agent.Health.CurrentHealth;

                // Target and relationship
                Vector3 targetPosition = Vector3.zero;
                Vector3 targetRotation = Vector3.zero;
                Vector3 targetForward = Vector3.zero;
                Vector3 directionToTarget = Vector3.zero;
                Vector3 cross = Vector3.zero;
                float distance = 0f;
                float dotProduct = 0f;
                bool islos = false;

                if (currentTargetAgentObject != null)
                {
                    targetPosition = currentTargetAgentObject.transform.position;
                    targetRotation = currentTargetAgentObject.transform.eulerAngles;
                    targetForward = currentTargetAgentObject.transform.forward;
                    distance = Vector3.Distance(agent.transform.position, currentTargetAgentObject.transform.position);

                    Vector3 directionWorld = (targetPosition - agentPosition).normalized;
                    directionToTarget = directionWorld;
                    dotProduct = Vector3.Dot(agentForward, directionWorld);
                    cross = Vector3.Cross(agentForward, directionWorld);

                    if (aiController != null)
                    {
                        islos = aiController.CheckLineOfSight2(gameObject, currentTargetAgentObject, 180);
                    }
                }

                return new ILState
                {
                    agentPosition = agentPosition,
                    agentRotation = agentRotation,
                    agentForward = agentForward,
                    health = health,
                    weapon = 0f,
                    targetPosition = targetPosition,
                    targetRotation = targetRotation,
                    targetForward = targetForward,
                    directionToTarget = directionToTarget,
                    cross = cross,
                    distance = distance,
                    dotProduct = dotProduct,
                    islos = islos
                };
            }
            catch (Exception e)
            {
                LogDebug($"Error creating current state: {e.Message}", true);
                return new ILState
                {
                    agentPosition = Vector3.zero,
                    agentRotation = Vector3.zero,
                    agentForward = Vector3.zero,
                    health = 0f,
                    weapon = 0f,
                    targetPosition = Vector3.zero,
                    targetRotation = Vector3.zero,
                    targetForward = Vector3.zero,
                    directionToTarget = Vector3.zero,
                    cross = Vector3.zero,
                    distance = 0f,
                    dotProduct = 0f,
                    islos = false
                };
            }
        }
        
        private async Task SendStateToAPI(ILState state)
        {
            if (isAPICallInProgress) return;
            
            isAPICallInProgress = true;
            lastAPICallTime = Time.time;
            
            try
            {
                var payload = new ILStatePayload { state = state };
                var json = JsonUtility.ToJson(payload, true);
                
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                LogDebug("Sending API request with single state");
                
                var response = await httpClient.PostAsync(apiUrl, content);
                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync();
                    LogDebug($"API Response: {responseContent}");
                    
                    // Deserialize the response
                    var apiResponse = JsonUtility.FromJson<ILAPIResponse>(responseContent);
                    if (apiResponse?.predictions != null && apiResponse.predictions.Length >= 2)
                    {
                        cachedPredictions = apiResponse.predictions;
                        ProcessAPIResponse();
                        LogDebug($"Updated predictions: [{string.Join(", ", cachedPredictions)}]");
                    }
                    else
                    {
                        LogDebug("Invalid API response format", true);
                    }
                }
                else
                {
                    LogDebug($"API Error: {response.StatusCode} - {response.ReasonPhrase}", true);
                    UseDefaultPredictions();
                }
            }
            catch (Exception e)
            {
                LogDebug($"Error sending states to API: {e.Message}", true);
                UseDefaultPredictions();
            }
            finally
            {
                isAPICallInProgress = false;
            }
        }
        
        private void ProcessAPIResponse()
        {
            try
            {
                // Server returns two floats: delta_x and delta_y
                float deltaX = cachedPredictions[0];
                float deltaY = cachedPredictions[1];
                // float deltaX = 0;
                // float deltaY = 1;
                // Interpret as world-space deltas to apply to current position to get the desired world target
                if (agent != null)
                {
                    Vector3 currentPos = agent.transform.position;
                    _targetWorldPosition = new Vector3(currentPos.x + deltaX, currentPos.y, currentPos.z + deltaY);
                    _hasWorldTarget = true;
                    if (debugLogMovement) GlipDebug.Log($"[ILInput] current world coordinates - {currentPos} New world target {_targetWorldPosition} from deltas dx:{deltaX}, dz:{deltaY}");
                }

                // No look or attack outputs from the model now
                AILookRotationDelta = Vector2.zero;
                AIShouldAttack = false;

                LogDebug($"Processed AI Input - World target set: {_targetWorldPosition}");
            }
            catch (Exception e)
            {
                LogDebug($"Error processing API response: {e.Message}", true);
                UseDefaultPredictions();
            }
        }
        
        private void UseDefaultPredictions()
        {
            // Use safe default values when API fails
            AIMovementDirection = Vector2.zero;
            AILookRotationDelta = Vector2.zero;
            AIShouldAttack = false;
            _hasWorldTarget = false;
            
            LogDebug("Using default predictions due to API failure");
        }
        
        private void LogDebug(string message, bool isError = false)
        {
            if (!enableDebugLogs && !isError) return;
            
            string prefix = $"[ILInput] ";
            // Route all logs through GlipDebug
            if (isError)
            {
                GlipDebug.Log(prefix + "[ERROR] " + message);
            }
            else
            {
                GlipDebug.Log(prefix + message);
            }
        }
        
        // Public methods for AgentInput to get AI predictions
        public Vector2 GetAIMovementDirection()
        {
            return AIMovementDirection;
        }
        
        public Vector2 GetAILookRotationDelta()
        {
            return AILookRotationDelta;
        }
        
        public bool GetAIShouldAttack()
        {
            return AIShouldAttack;
        }
        
        // Method to check if API is available and responding
        public bool IsAPIResponding()
        {
            return !isAPICallInProgress && (Time.time - lastAPICallTime) < (apiCallInterval * 2);
        }
        
        void OnDestroy()
        {
            httpClient?.Dispose();
        }
    }
}
