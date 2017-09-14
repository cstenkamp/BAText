input metauml;

iClass.iName.iFont.scale := 1.1;
iAssoc.iFont.scale := 1.2;

AbstractClass.agent("AbstractAgent")
  ("-action_repeat: int=4",
  "-isSupervised: bool=False",
  "-isContinuous: bool=False",
  "-usesConv: bool=True",
  "+usesGUI: bool=False",
  "-evaluator: evaluator(settings)",
   "-model: None")
  ("__init__",
   "checkIfAction(): bool",
   "-getAgentState(gameState)",
   "makeNetUsableOtherInputs",
   "getAction",
   "makeNetUsableAction",
   "makeInferenceUsable",
   "+initForDriving(*args)",
   "+performAction(gameState,pastState)",
   "dediscretize",
   "discretize",
   "inflate_spped",
   "-resetUnityAndServer()",
   "folder",
   "+handle_commands(command: string)");


AbstractClass.rlagent("AbstractRLAgent")
  ("-start_fresh: bool",
   "-wallhitPunish: int=1",
   "-wrongDirPunish: int=5",
   "+time_ends_episode: int=60",
   "show_plots: bool",
   "-startepsilon: float",
   "minepsilon: float",
   "finalepsilonframe: int",
   "-epsilon: float",
   "-evaluator: evaluator(settings)",
   "-memory: Inefficientmemory")
  ("__init__",
   "policyAction",
   "+initForDriving(*args)",
   "-calculateReward(gameState)",
   "-randomAction(agentState)",
   "+performAction(gameState,pastState)",
   "-addToMemory(gameState,pastState)",
   "+handle_commands(command: string)",
   "checkIfAction",
   "canLearn",
   "+dauerLearnANN(steps: int)",
   "-learnANN()",
   "-punishLastAction(howmuch: int)",
   "-endEpisode(reason: string, gameState)",
   "saveNet",
   "eval_episodeVals",
   "create_QLearnInputs_from_MemoryBatch",
   "+preTrain(dataset, iterations: int)",
   "freeze-functions");


Class.dqnagent("dqn_rl_agent.Agent")
  ("ff_inputsize: int=32",
   "session: tf.Session",
   "model: DDDQN_model",
   "usesGUI: bool=True",
   "memory: EfficientMemory")
  ("policyAction",
   "randomAction");

Class.blindqnagent("dqn_novision_rl_agent.Agent")
  ("ff_inputsize: int=156",
   "session: tf.Session",
   "usesConv: bool=False",
   "model: DDDQN_model",
   "usesGUI: bool=True",
   "memory: InefficientMemory")
  ("getAgentState",
   "makeNetUsableOtherInputs",
   "policyAction",
   "randomAction");


Class.blinddpgagent("ddpg_novision_rl_agent.Agent")
  ("ff_inputsize: int=156",
   "session: tf.Session",
   "isContinuous: bool=True",
   "usesConv: bool=False",
   "usesGUI: bool=True",
   "-_noiseState: [float]",
   "model: DDPG_model",
   "memory: InefficientMemory",
   "-_noiseState: [float]")
  ("makeNetUsableAction",
   "getAgentState",
   "makeNetUsableOtherInputs",
   "make_noisy",
   "-endEpisode",
   "randomAction",
   "policyAction",
   "preTrain");


Class.ddpgagent("ddpg_rl_agent.Agent")
("ff_inputsize: int=32",
"model: DDPG_model",
"usesGUI: bool=True",
"isContinous: bool=True",
"memory: EfficientMemory",
"-_noiseState: [float]")
("initForDriving",
"makeNetUsableAction",
"-make_noisy",
"-endEpisode",
"randomAction",
"policyAction",
"preTrain"); 

   
Class.svagent("dqn_sv_agent.Agent")
("ff_inputsize: int=32",
"model: DDDQN_model",
"isSupervised: bool=True")
("policyAction",
"preTrain");    


Class.dddqn("DDDQN_model")
  ("onlineQN: DuelDQN", "targetQN: DuelDQN")
  ("getAccuracy()", "inference()", "statevalue()", "sv_learn()", "q_learn()");

Class.ememory("EfficientMemory")
  ("len")
  ("add");

Class.ddpg("DDPG_model")
  ("onlineQN: DuelDQN", "targetQN: DuelDQN")
  ("getAccuracy()", "inference()", "statevalue()", "sv_learn()", "q_learn()");


beginfig(1)

	topToBottom(50)(agent, rlagent, dqnagent);
	leftToRight.top(20)(blindqnagent, dqnagent, blinddpgagent);
	topToBottom(100)(dqnagent, ememory);
	leftToRight.top(80)(dddqn, ememory, ddpg);
	

	drawObjects(agent, rlagent, blindqnagent, dqnagent, blinddpgagent, ddpg, dddqn, ememory);


	link(inheritance)(rlagent.n -- agent.s);
	link(inheritance)(pathStepY(dqnagent.n, rlagent.s, 15));
	link(inheritance)(pathStepY(blindqnagent.n, rlagent.s, 15));
	link(inheritance)(pathStepY(blinddpgagent.n, rlagent.s, 15));

	clink(composition)(dqnagent,dddqn);
	clink(composition)(blindqnagent,dddqn);
	clink(composition)(blindqnagent,ememory);
	clink(composition)(blinddpgagent,ememory);
	clink(composition)(blinddpgagent,ddpg);
	clink(composition)(dqnagent,ememory);

endfig;


end
