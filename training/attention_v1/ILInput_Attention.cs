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
    public class AgentFeatures
    {
        public float team_index;
        public float rel_pos_x;
        public float rel_pos_z;
        public float rotation;
        public float move_dir_x;
        public float move_dir_y;
        public float look_rot_delta_x;
        public float look_rot_delta_y;
        public float attack;
        public float shrinking_key;
        public float delta_x;
        public float delta_y;
        public float delta_rot;
    }

    [System.Serializable]
    public class UnityTemporalState
    {
        public List<AgentFeatures> agents;
    }

    [System.Serializable]
    public class UnitySpatialState
    {
        public List<AgentFeatures> agents;
    }

    [System.Serializable]
    public class UnityStatePayload
    {
        public UnityTemporalState temporal;
        public UnitySpatialState spatial;
    }

    [System.Serializable]
    public class UnityDualOutputResponse
    {
        public float[] predictions;
    }

    public class ILInput_Attention : MonoBehaviour
    {
        [Header("API Configuration")]
        [SerializeField] private string apiUrl = "http://localhost:8000/predict/unity";
        [SerializeField] private bool enableAPIInput = true;
        [SerializeField] private float apiCallInterval = 0.05f; // Call API every 50ms

        [Header("Debug")]
        [SerializeField] private bool enableDebugLogs = true;

        [Header("Feature Collection")]
        [SerializeField] private int maxTemporalHistory = 64; // Maximum history length
        [SerializeField] private float featureUpdateInterval = 0.1f; // How often to update features

        // Core components
        private Agent agent;
        private GameObject currentTargetAgentObject;
        private AIController aiController;

        // API communication
        private HttpClient httpClient = new HttpClient();
        private float[] cachedPredictions = new float[2] { 0f, 0f };

        // Timing
        private float lastAPICallTime = 0f;
        private float lastFeatureUpdateTime = 0f;
        private bool isAPICallInProgress = false;

        // Output values for AgentInput
        public Vector2 AIMovementDirection { get; private set; } = Vector2.zero;
        public Vector2 AILookRotationDelta { get; private set; } = Vector2.zero;
        public bool AIShouldAttack { get; private set; } = false;

        // World-space movement target derived from server deltas
        private bool _hasWorldTarget = false;
        private Vector3 _targetWorldPosition = Vector3.zero;
        [SerializeField] private float targetReachThreshold = 0.25f;
        [SerializeField] private bool debugLogMovement = true;
        [SerializeField] private float arriveBrakingDistance = 3.0f;
        [SerializeField] private bool snapAtGoal = false;
        [SerializeField] private float precisionSnapThreshold = 0.01f;

        // Feature history storage
        private List<AgentFeatures> temporalHistory = new List<AgentFeatures>();
        private AgentFeatures lastAgentFeatures;
        private Vector3 lastAgentPosition;
        private float lastUpdateTime;

        // Running average for movement inputs
        [SerializeField] private int movementAverageWindow = 20;
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

            // Update features periodically
            if (Time.time - lastFeatureUpdateTime >= featureUpdateInterval)
            {
                UpdateAgentFeatures();
                lastFeatureUpdateTime = Time.time;
            }

            // Drive movement toward world target
            if (_hasWorldTarget)
            {
                Vector3 currentPos = agent.transform.position;
                Vector3 toTarget = _targetWorldPosition - currentPos;
                float distance = toTarget.magnitude;

                // Optional precise snap when extremely close
                if (snapAtGoal && distance <= precisionSnapThreshold)
                {
                    var p = agent.transform.position;
                    agent.transform.position = new Vector3(_targetWorldPosition.x, p.y, _targetWorldPosition.z);
                    _hasWorldTarget = false;
                    AIMovementDirection = Vector2.zero;
                    if (debugLogMovement) GlipDebug.Log($"[ILInput_Attention] Snapped to target {_targetWorldPosition}");
                }
                else if (distance <= targetReachThreshold)
                {
                    _hasWorldTarget = false;
                    AIMovementDirection = Vector2.zero;
                    if (debugLogMovement) GlipDebug.Log($"[ILInput_Attention] Reached target {_targetWorldPosition}");
                }
                else
                {
                    Vector3 worldDir = toTarget;
                    Vector3 localDir = agent.transform.InverseTransformDirection(worldDir);
                    float scale = Mathf.Clamp01(distance / Mathf.Max(arriveBrakingDistance, 0.0001f));
                    Vector2 inputDir = new Vector2(localDir.x, localDir.z);
                    AIMovementDirection = GetAveragedMovement(inputDir * 5);
                    if (debugLogMovement) GlipDebug.Log($"[ILInput_Attention] Moving toward {_targetWorldPosition}, distance {distance:F2}, input {AIMovementDirection}");
                }
            }

            // Check if it's time to make an API call
            if (Time.time - lastAPICallTime >= apiCallInterval && !isAPICallInProgress)
            {
                var state = CreateCurrentState();
                _ = SendStateToAPI(state);
            }
        }

        private void UpdateAgentFeatures()
        {
            if (agent == null) return;

            // Calculate deltas from last frame
            Vector3 currentPos = agent.transform.position;
            float deltaTime = Time.time - lastUpdateTime;

            float delta_x = 0f;
            float delta_y = 0f;
            if (lastAgentPosition != Vector3.zero && deltaTime > 0)
            {
                Vector3 deltaPos = currentPos - lastAgentPosition;
                delta_x = deltaPos.x;
                delta_y = deltaPos.z; // Unity's Z is our Y
            }

            // Create current agent features
            var features = CreateAgentFeatures(agent, delta_x, delta_y, 0f); // No rotation delta for now

            // Add to temporal history
            temporalHistory.Add(features);
            if (temporalHistory.Count > maxTemporalHistory)
            {
                temporalHistory.RemoveAt(0); // Remove oldest
            }

            // Update tracking variables
            lastAgentFeatures = features;
            lastAgentPosition = currentPos;
            lastUpdateTime = Time.time;
        }

        private AgentFeatures CreateAgentFeatures(Agent agent, float delta_x, float delta_y, float delta_rot)
        {
            if (agent == null)
            {
                return new AgentFeatures
                {
                    team_index = 0f,
                    rel_pos_x = 0f,
                    rel_pos_z = 0f,
                    rotation = 0f,
                    move_dir_x = 0f,
                    move_dir_y = 0f,
                    look_rot_delta_x = 0f,
                    look_rot_delta_y = 0f,
                    attack = 0f,
                    shrinking_key = 0f,
                    delta_x = 0f,
                    delta_y = 0f,
                    delta_rot = 0f
                };
            }

            try
            {
                // Get target information
                currentTargetAgentObject = agent.ReturnTargetAgent(agent, 100000.0f);
                Vector3 targetPosition = Vector3.zero;
                Vector3 targetRotation = Vector3.zero;
                Vector3 targetForward = Vector3.zero;

                if (currentTargetAgentObject != null)
                {
                    targetPosition = currentTargetAgentObject.transform.position;
                    targetRotation = currentTargetAgentObject.transform.eulerAngles;
                    targetForward = currentTargetAgentObject.transform.forward;
                }

                // Calculate relative position (normalized)
                Vector3 relativePos = targetPosition - agent.transform.position;
                float rel_pos_x = relativePos.x * 0.01f; // Scale down
                float rel_pos_z = relativePos.z * 0.01f;

                // Agent's rotation (normalized to 0-1)
                float rotation = agent.transform.eulerAngles.y / 360f;

                // Movement direction (from last frame delta, normalized)
                float move_dir_x = Mathf.Clamp(delta_x * 10f, -1f, 1f);
                float move_dir_y = Mathf.Clamp(delta_y * 10f, -1f, 1f);

                // Look rotation delta (simplified - could be improved)
                float look_rot_delta_x = 0f;
                float look_rot_delta_y = 0f;

                // Attack state (simplified)
                float attack = agent.AgentInput.FixedInput.Attack ? 1f : 0f;

                // Shrinking key (placeholder - would need game-specific implementation)
                float shrinking_key = 0f;

                return new AgentFeatures
                {
                    team_index = 0f, // Would need team identification logic
                    rel_pos_x = rel_pos_x,
                    rel_pos_z = rel_pos_z,
                    rotation = rotation,
                    move_dir_x = move_dir_x,
                    move_dir_y = move_dir_y,
                    look_rot_delta_x = look_rot_delta_x,
                    look_rot_delta_y = look_rot_delta_y,
                    attack = attack,
                    shrinking_key = shrinking_key,
                    delta_x = delta_x,
                    delta_y = delta_y,
                    delta_rot = delta_rot
                };
            }
            catch (Exception e)
            {
                LogDebug($"Error creating agent features: {e.Message}", true);
                return new AgentFeatures
                {
                    team_index = 0f,
                    rel_pos_x = 0f,
                    rel_pos_z = 0f,
                    rotation = 0f,
                    move_dir_x = 0f,
                    move_dir_y = 0f,
                    look_rot_delta_x = 0f,
                    look_rot_delta_y = 0f,
                    attack = 0f,
                    shrinking_key = 0f,
                    delta_x = 0f,
                    delta_y = 0f,
                    delta_rot = 0f
                };
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

            // Only run ILInput_Attention for the local player (has input authority) and skip AI bots
            if (agent.Object == null || !agent.Object.HasInputAuthority || agent.isAIBot)
            {
                LogDebug("ILInput_Attention disabled because the agent either lacks input authority or is a bot.");
                enabled = false;
                return;
            }

            // Initialize tracking variables
            lastAgentPosition = agent.transform.position;
            lastUpdateTime = Time.time;

            LogDebug("ILInput_Attention initialized successfully");
        }

        private UnityStatePayload CreateCurrentState()
        {
            if (agent == null)
            {
                return new UnityStatePayload
                {
                    temporal = new UnityTemporalState { agents = new List<AgentFeatures>() },
                    spatial = new UnitySpatialState { agents = new List<AgentFeatures>() }
                };
            }

            try
            {
                // Create temporal state (agent's history)
                var temporalState = new UnityTemporalState
                {
                    agents = new List<AgentFeatures>(temporalHistory)
                };

                // Create spatial state (current snapshot of all agents)
                var spatialAgents = new List<AgentFeatures>();

                // Add the current agent
                if (lastAgentFeatures != null)
                {
                    spatialAgents.Add(lastAgentFeatures);
                }

                // Add target agent if exists
                if (currentTargetAgentObject != null)
                {
                    var targetAgent = currentTargetAgentObject.GetComponent<Agent>();
                    if (targetAgent != null)
                    {
                        // Create features for target agent
                        var targetFeatures = CreateAgentFeatures(targetAgent, 0f, 0f, 0f);
                        spatialAgents.Add(targetFeatures);
                    }
                }

                // TODO: Add all other agents in the scene for complete spatial snapshot
                // This would require finding all Agent objects in the scene

                var spatialState = new UnitySpatialState
                {
                    agents = spatialAgents
                };

                return new UnityStatePayload
                {
                    temporal = temporalState,
                    spatial = spatialState
                };
            }
            catch (Exception e)
            {
                LogDebug($"Error creating current state: {e.Message}", true);
                return new UnityStatePayload
                {
                    temporal = new UnityTemporalState { agents = new List<AgentFeatures>() },
                    spatial = new UnitySpatialState { agents = new List<AgentFeatures>() }
                };
            }
        }

        private async Task SendStateToAPI(UnityStatePayload state)
        {
            if (isAPICallInProgress) return;

            isAPICallInProgress = true;
            lastAPICallTime = Time.time;

            try
            {
                var json = JsonUtility.ToJson(state, true);

                var content = new StringContent(json, Encoding.UTF8, "application/json");

                LogDebug("Sending API request with temporal and spatial states");

                var response = await httpClient.PostAsync(apiUrl, content);
                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync();
                    LogDebug($"API Response: {responseContent}");

                    // Deserialize the response
                    var apiResponse = JsonUtility.FromJson<UnityDualOutputResponse>(responseContent);
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

                // Interpret as world-space deltas to apply to current position to get the desired world target
                if (agent != null)
                {
                    Vector3 currentPos = agent.transform.position;
                    _targetWorldPosition = new Vector3(currentPos.x + deltaX, currentPos.y, currentPos.z + deltaY);
                    _hasWorldTarget = true;
                    if (debugLogMovement) GlipDebug.Log($"[ILInput_Attention] current world coordinates - {currentPos} New world target {_targetWorldPosition} from deltas dx:{deltaX}, dz:{deltaY}");
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

            string prefix = $"[ILInput_Attention] ";
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

        // Method to reset feature history
        public void ResetFeatureHistory()
        {
            temporalHistory.Clear();
            lastAgentFeatures = null;
            lastAgentPosition = agent ? agent.transform.position : Vector3.zero;
            lastUpdateTime = Time.time;
            LogDebug("Feature history reset");
        }

        void OnDestroy()
        {
            httpClient?.Dispose();
        }
    }
}
