using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System;
using System.Collections.Generic;
using TPSBR.Core;
using TPSBR.UI;
using ExitGames.Client.Photon.StructWrapping;
using Dissonance;
using UnityEngine.EventSystems;
using Fusion.KCC;
using CrashKonijn.Goap.Classes.Runners;
using UnityEngine.InputSystem;
using Object = UnityEngine.Object;
using System.Collections;
using UnityEngine.Serialization;
using System.Collections;
using Fusion;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace TPSBR
{
    	// [System.Serializable]
        // public class LogValues
        // {
        //     public string game_id;
        //     public double game_time;
        //     public int agent_id;
        //     public Vector3 agentPosition;
        //     public Vector3 agentRotation;
        //     public Vector3 targetPosition;
        //     public float distance;
        //     public float dotProduct;
        //     public bool islos;
        //     public string timestamp;
        //     public int[] discrete_actions;  // Array to store all discrete actions
        // }

    [System.Serializable]
    public class State
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
    public class StatePayload
    {
        public List<State> states;
    }

    [System.Serializable]
    public class APIResponse
    {
        public float[] predictions;
    }

    public class MLAgent : Unity.MLAgents.Agent
    {
        public Agent agent;
        public GameObject currentTargetAgentObject;
        public Health currentTargetAgentHealth;
        public Agent currentTargetAgent;
        private float episodeStartTime;  
        public AIController aiController;
        private float previousDamageGiven;
        public UnityEngine.UI.Text observationText; // Assign in Inspector
        private bool los;
        private static System.Random rand = new System.Random();
        [SerializeField] private RayPerceptionSensorComponent3D raySensorComponent;
        private GameObject lastDetectedPlayer = null;  // Reference to the last detected "Player"
        private bool playerDetected = false;           // Flag to ensure one-time detection
        private bool wallDetected = false;
        
        private int bufferIndex = 0;                     // Current position in the buffer
        
        private bool[] detectionBufferM = new bool[100]; // Buffer for SphereM detections
        
        private float[] distanceBuffer = new float[100];           // Buffer for distances to target
        
        // Reference to the RayPerceptionSensor from the "Sphere" child
        private RayPerceptionSensor m_RaySensor;
        private float smoothing_factor_move = 0.1f;  // Smoothing factor for movement (0.0-1.0)
        private float smoothing_factor_turn = 0.05f; // Smoothing factor for turning (0.0-1.0)
        
        // Smoothed values for movement and look inputs
        private float smoothed_move_x = 0f;
        private float smoothed_move_z = 0f;
        private float smoothed_turn = 0f;

        
		private string _logFolderPath;
		private string _logFilePath;
		private StreamWriter _logWriter;

        private float previousDistanceToTarget = float.MaxValue;  // Add this line to track previous distance
        		
        // Add new field declarations for cached observations
        private Vector3 _cachedAgentPosition;
        private Vector3 _cachedAgentRotation;
        private Vector3 _cachedTargetPosition;
        private float _cachedDistance;
        private float _cachedDotProduct;
        private bool _cachedLineOfSight;

        // Add new field declarations for logging
        private string gameID;
        private int agentID;
        private GameObject _character;

        private ConcurrentQueue<State> stateHistory = new ConcurrentQueue<State>();
        private const int MAX_STATES = 20;
        private HttpClient httpClient = new HttpClient();
        private string apiUrl = "http://localhost:8000/predict";
        private float[] cachedPredictions = new float[7] { 0f, 0f, 0f, 0f, 0f, 0f, 0f };  // Array to cache predictions
        private float[]cachedmovedirection = new float[2] { 0, 0 };
        private List<float> cachedfeatures = new List<float>();

        private void InitializeLogger()
        {
            string game_path = Path.Combine("gamelogs/", gameID);
            _logFolderPath = Path.Combine(Application.persistentDataPath, game_path );
            Directory.CreateDirectory(_logFolderPath);
            Debug.Log($"Initializing logger with folder path: {_logFolderPath}");
            // Create unique filename
            string filename = $"{gameID}_Agent_{agentID}.json";
            _logFilePath = Path.Combine(_logFolderPath, filename);

            // Initialize writer
            _logWriter = new StreamWriter(_logFilePath, true);
            _logWriter.AutoFlush = true;
        }
        // private void LogTransformData(float game_time)
        // {
        //     if (_logWriter == null)
        //     	InitializeLogger();

        //     var data = new LogValues
        //     {
        //         game_id = gameID,
        //         game_time = game_time,
        //         agent_id = agentID,
        //         agentPosition = _cachedAgentPosition,
        //         agentRotation = _cachedAgentRotation,
        //         targetPosition = _cachedTargetPosition,
        //         distance = _cachedDistance,
        //         dotProduct = _cachedDotProduct,
        //         islos = _cachedLineOfSight,
        //         timestamp = System.DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        //     };

        //     string json = JsonUtility.ToJson(data, true);
        //     _logWriter.WriteLine(json);
        // }
        void Start()
        {
            GlipDebug.Log("MLAgent - Start");
            agent = GetComponent<Agent>();
            if (agent == null)
            {
                throw new Exception("Agent component not found on this GameObject.");
            }

            // Set the character reference
            _character = gameObject;

            // Disable NavMeshAgent for ML-controlled agents to prevent position conflicts
            var navMeshAgent = GetComponent<UnityEngine.AI.NavMeshAgent>();
            if (navMeshAgent != null)
            {
                navMeshAgent.enabled = false;
                GlipDebug.Log("MLAgent - Disabled NavMeshAgent to prevent movement conflicts");
            }

            // if (!agent.isAIBot)
            // {
            //     this.enabled = false;
            //     return;
            // }
            // Initialize cached observation values
            _cachedAgentPosition = Vector3.zero;
            _cachedAgentRotation = Vector3.zero;
            _cachedTargetPosition = Vector3.zero;
            _cachedDistance = 0f;
            _cachedDotProduct = 0f;
            _cachedLineOfSight = false;

            // Initialize logging values
            gameID = System.Guid.NewGuid().ToString();
            agentID = 0; // This will be set properly when the agent is created

            if(agent.isAIBot)
            {
                agent.Weapons.SwitchToNextWeapon();
            }
        }

        public override void OnEpisodeBegin()
        {
            //GlipDebug.Log("MLAgent - OnEpisodeBegin start");
            episodeStartTime = Time.time;
            playerDetected = false;
            lastDetectedPlayer = null;
            previousDistanceToTarget = float.MaxValue;  
            
            // Reset smoothed values
            smoothed_move_x = 0f;
            smoothed_move_z = 0f;
            smoothed_turn = 0f;
            
            // Reset detection buffers
            for (int i = 0; i < 100; i++)
            {
                distanceBuffer[i] = float.MaxValue;     // Initialize distance buffer to max value
            }
            bufferIndex = 0;
            
            try
            {
                // Print the agent's user ID
                if (agent != null && agent.BtxPlayerRef.IsValid)
                {
                    //GlipDebug.Log("MLAgent Episode Begin - Agent UserID: " + agent.BtxPlayerRef.userId);
                }
                agent = GetComponent<Agent>();
                
                if (agent == null)
                {
                    throw new Exception("Agent component not found on this GameObject.");
                }

               
                // if (agent.isAIBot)
                // {
                //     //GlipDebug.Log("MLbot episode begin : not ai bot");
                //     this.enabled = false;
                //     return;
                // }
               
                // else{
                //     this.enabled = true;
                // }
            }
            catch (Exception e)
            {
                //GlipDebug.Log("Episode " + e);
                this.enabled = false;
            }
        }

        // Update is called once per frame
        public override void CollectObservations(VectorSensor sensor)
        {
            GlipDebug.Log("MLAgent - CollectObservations start");

            if (agent == null)
            {
                throw new Exception("Agent component not found on this GameObject.");
            }
            cachedfeatures.Clear();
            currentTargetAgentObject = agent.ReturnTargetAgent(agent, 100000.0f);
            
            Vector3 agentPosition = agent.transform.position/1000;
            Vector3 agentRotation = agent.transform.eulerAngles/360;
            Vector3 agentforward = agent.transform.forward;
            float health = agent.Health.CurrentHealth/100;
            Vector3 targetPosition = Vector3.zero;
            Vector3 targetRotation = Vector3.zero;
            float distance = 0;
            Vector3 directionToTarget = Vector3.zero;
            Vector3 agentForward = agent.transform.forward;
            Vector3 targetForward = Vector3.zero;
            float dotProduct = 0;
            bool islos = false;
            Vector3 cross = Vector3.zero;

            if (currentTargetAgentObject != null) {
                targetPosition = currentTargetAgentObject.transform.position/1000;
                targetRotation = currentTargetAgentObject.transform.eulerAngles/360;
                distance = Vector3.Distance(agent.transform.position, currentTargetAgentObject.transform.position)/1000;
                directionToTarget = (targetPosition - agentPosition).normalized;
                targetForward = currentTargetAgentObject.transform.forward;
                dotProduct = Vector3.Dot(agentForward, directionToTarget);
                AIController agentAIController = GetComponent<AIController>();
                islos = agentAIController.CheckLineOfSight2(this.gameObject, currentTargetAgentObject, 180);
                cross = Vector3.Cross(agentForward, directionToTarget);
            }

            cachedfeatures.Add(agentPosition.x);
            cachedfeatures.Add(agentPosition.y); 
            cachedfeatures.Add(agentPosition.z);
            cachedfeatures.Add(agentRotation.y);
            cachedfeatures.Add(agentforward.x);
            cachedfeatures.Add(agentforward.y);
            cachedfeatures.Add(agentforward.z);
            cachedfeatures.Add(health);
            cachedfeatures.Add(0); // weapon 
            cachedfeatures.Add(islos ? 1.0f : 0.0f);
            cachedfeatures.Add(targetPosition.x);
            cachedfeatures.Add(targetPosition.y);
            cachedfeatures.Add(targetPosition.z);
            cachedfeatures.Add(targetRotation.y);
            cachedfeatures.Add(targetForward.x);
            cachedfeatures.Add(targetForward.y);
            cachedfeatures.Add(targetForward.z);
            cachedfeatures.Add(directionToTarget.x);
            cachedfeatures.Add(directionToTarget.y);
            cachedfeatures.Add(directionToTarget.z);
            cachedfeatures.Add(cross.x);
            cachedfeatures.Add(cross.y);
            cachedfeatures.Add(cross.z);
            cachedfeatures.Add(distance);
            cachedfeatures.Add(dotProduct);


            GlipDebug.Log("MLAgent - cached features: " + string.Join(", ", cachedfeatures));

        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // var discreteActions = actionsOut.DiscreteActions;
            // var keyboard = Keyboard.current;
            
            // var mouse = Mouse.current;
            
            //     // Fallback to original keyboard input if AgentInput not found
            //     discreteActions[0] = keyboard.wKey.isPressed ? 1 : 0;
            //     discreteActions[1] = keyboard.sKey.isPressed ? 1 : 0;
            //     discreteActions[2] = keyboard.aKey.isPressed ? 1 : 0;
            //     discreteActions[3] = keyboard.dKey.isPressed ? 1 : 0;
                
            //     var mouseDelta = mouse.delta.ReadValue() * 0.075f;
            //     if(mouseDelta.x > 1)
            //     {
            //         discreteActions[4] = 2;
            //     }
            //     else if(mouseDelta.x < -1)
            //     {
            //         discreteActions[4] = 1;
            //     }
            //     else 
            //     {
            //         discreteActions[4] = 0;
            //     }
            //     discreteActions[5] = mouse.leftButton.isPressed ? 1 : 0;
            


            
            // if (agent == null)
            // {
            //     throw new Exception("Agent component not found on this GameObject.");
            // }

            // currentTargetAgentObject = agent.ReturnTargetAgent(agent, 100000.0f);
            // // try{
            // //     if (agent.isAIBot)
            // //     {
            // //         GlipDebug.Log("Agent is an AI bot. Exiting CollectObservations early.");
            // //         return;
            // //     }
            // // }
            // // catch(Exception e){
            // //     //GlipDebug.Log("Exception in CollectObservations: " + e);
            // //     return;
            // // }



            // // Log the observation data
            // if (_logWriter == null)
            //     InitializeLogger();

            // var data = new LogValues
            // {
            //     game_id = agent.Runner.SessionInfo.Name,
            //     game_time = agent.Runner.SimulationTime,
            //     agent_id = agent.Object.InputAuthority.PlayerId,
            //     agentPosition = _cachedAgentPosition,
            //     agentRotation = _cachedAgentRotation,
            //     targetPosition = _cachedTargetPosition,
            //     distance = _cachedDistance,
            //     dotProduct = _cachedDotProduct,
            //     islos = _cachedLineOfSight,
            //     timestamp = System.DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
            //     discrete_actions = new int[6]  // Initialize array for 6 discrete actions
            // };

            // // Copy all discrete actions to the array
            // for (int i = 0; i < 6; i++)
            // {
            //     data.discrete_actions[i] = actionsOut.DiscreteActions[i];
            // }

            // string json = JsonUtility.ToJson(data, true);
            // _logWriter.WriteLine(json);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
           
            // if (!agent.isAIBot)
            // {
            //     //GlipDebug.Log("Agent is not an AI bot. Exiting OnActionsReceived early.");
            //     return;
            // }
             GlipDebug.Log("MLAgent - on action received working fine !");
             // Add current state to history
            var currentState = CreateCurrentState();
            stateHistory.Enqueue(currentState);

            // Maintain queue size
            while (stateHistory.Count > MAX_STATES)
            {
                stateHistory.TryDequeue(out _);
            }
            bufferIndex++;
            // if(bufferIndex % 10 ==0)
            // {
            //     _= GenerateRandomActions();
            // }
            _= SendStatesToAPI();
            GlipDebug.Log($"MLAgent - cachedPredictions values : [{string.Join(", ", cachedPredictions)}]");

            Vector2 MoveDirection = new Vector2(0,0);
            MoveDirection += new Vector2(cachedPredictions[1], cachedPredictions[0]); 
            
            Vector3 directionToTarget = (_cachedTargetPosition - _cachedAgentPosition).normalized;
            
            // Get agent's forward direction
            Vector3 agentForward = agent.transform.forward;
            Vector3 cross = Vector3.Cross(agentForward, directionToTarget);
            float targetSide = cross.y; // Positive means target is to the left, negative means target is to the right
            
            Vector2 mouseDelta = new Vector2(0f, 0f);

            // mouseDelta += new Vector2(cachedPredictions[2], 0);
            
            
            float pit = agent.get_aim_pitch();
            if (pit > 1)
            {
                mouseDelta += new Vector2(0, 1.0f);
            }
            if (pit < -1)
            {
                mouseDelta += new Vector2(0, -1.0f);
            }
            bool shoot = false;

            // Convert float prediction to boolean for shooting action
            // You can adjust the threshold (0.5f) based on your needs
            shoot = cachedPredictions[3] > 0.5f;

            // switch (cachedPredictions[6])
            // {
            //     case 1: shoot = true; break;
            // }
            // if(cachedPredictions[6] ==1)
            // {
            //     shoot = true;
            // }
            // else if(_cachedDotProduct>0.99 && _cachedLineOfSight)
            // {
            //     shoot = true;
            // }
            // else
            // {
            //     shoot = false;
            // }

            // Add detailed debug logging
            GlipDebug.Log($"MLAgent - OnActionReceived Debug: MoveDirection={MoveDirection}, mouseDelta={mouseDelta}, shoot={shoot}");
            GlipDebug.Log($"MLAgent - Agent Details: position={agent.transform.position}, isAIBot={agent.isAIBot}, hasStateAuthority={agent.Object.HasStateAuthority}");


            GlipDebug.Log("MLAgent - setting cached inputs");
            
            // Enable sprint when moving to ensure proper movement speed
            bool shouldSprint = MoveDirection.magnitude > 0.1f;
            
            agent.SetAICachedInputs(
                MoveDirection,     // Movement direction
                mouseDelta,        // Look rotation delta
                false,             // Jump
                shoot,             // Attack
                false,             // Reload
                false,             // Interact
                0,                 // Weapon
                shouldSprint,      // Sprint - enable when moving
                false,             // TacticalSprint
                false,             // ToggleJetpack
                false,             // Thrust
                false              // Crouch
            );
            // if (agent.isAIBot)
            // {
            //         agent.SetAICachedInputs(MoveDirection, mouseDelta, false, shoot);
            // }
            

        }

        private State CreateCurrentState()
        {
            if (cachedfeatures.Count < 25) // Ensure we have enough features
            {
                return new State
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

            return new State
            {
                // Agent state variables (indices 0-9)
                agentPosition = new Vector3(cachedfeatures[0], cachedfeatures[1], cachedfeatures[2]),
                agentRotation = new Vector3(0, cachedfeatures[3], 0), // Only Y rotation is stored
                agentForward = new Vector3(cachedfeatures[4], cachedfeatures[5], cachedfeatures[6]),
                health = cachedfeatures[7],
                weapon = cachedfeatures[8],
                islos = cachedfeatures[9] > 0.5f,
                
                // Target state variables (indices 10-17)
                targetPosition = new Vector3(cachedfeatures[10], cachedfeatures[11], cachedfeatures[12]),
                targetRotation = new Vector3(0, cachedfeatures[13], 0), // Only Y rotation is stored
                targetForward = new Vector3(cachedfeatures[14], cachedfeatures[15], cachedfeatures[16]),
                
                // Relationship variables (indices 17-24)
                directionToTarget = new Vector3(cachedfeatures[17], cachedfeatures[18], cachedfeatures[19]),
                cross = new Vector3(cachedfeatures[20], cachedfeatures[21], cachedfeatures[22]),
                distance = cachedfeatures[23],
                dotProduct = cachedfeatures[24]
            };
        }

        private async Task GenerateRandomActions()
        {
            try
            {
                // Generate random predictions for 7 actions
                // Actions 0-3: Movement (forward, backward, left, right)
                // Action 4-5: Turning (left, right)
                // Action 6: Shooting
                for (int i = 0; i < 7; i++)
                {
                    // For movement and turning actions (0-5), generate random float between 0.0 and 1.0
                    // For shooting action (6), generate random float between 0.0 and 1.0
                    cachedPredictions[i] = (float)rand.NextDouble();
                }
                
                Debug.Log("Generated random predictions: " + string.Join(", ", cachedPredictions));
                
            }
            catch (Exception e)
            {
                Debug.LogError($"Error generating random actions: {e.Message}");
                
            }
        }

        private async Task SendStatesToAPI()
        {
            try
            {
                var states = new List<State>(stateHistory);
                
                // If we have less than MAX_STATES states, pad with sample states
                if (states.Count < MAX_STATES)
                {
                    int statesToAdd = MAX_STATES - states.Count;
                    for (int i = 0; i < statesToAdd; i++)
                    {
                        states.Add(new State
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
                        });
                    }
                }

                var payload = new StatePayload { states = states };
                var json = JsonUtility.ToJson(payload, true);

                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await httpClient.PostAsync(apiUrl, content);
                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync();
                    Debug.Log($"API Response: {responseContent}");
                    
                    // Deserialize the response
                    var apiResponse = JsonUtility.FromJson<APIResponse>(responseContent);
                    cachedPredictions = apiResponse.predictions;  // Update cached predictions
                    Debug.Log("Predictions: " + string.Join(", ", cachedPredictions));
                }
                else
                {
                    Debug.LogError($"API Error: {response.StatusCode}");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Error sending states to API: {e.Message}");
            }
        }

    }
}
