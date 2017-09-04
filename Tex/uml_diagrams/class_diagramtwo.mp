input metauml;

iClass.iName.iFont.scale := 1.1;
iAssoc.iFont.scale := 1.2;

AbstractClass.agent("AbstractAgent")
  ("isInitialized: bool",
   "action_repeat: int=4",
   "model: None")
  ("__init__",
   "checkIfAction(): bool",
   "getAgentState",
   "makeNetUsableOtherInputs",
   "getAction",
   "makeNetUsableAction",
   "makeInferenceUsable",
   "initForDriving",
   "performAction",
   "dediscretize",
   "discretize",
   "inflate_spped",
   "reesetUnityAndServer",
   "folder");


AbstractClass.rlagent("AbstractRLAgent")
  ("start_fresh: bool",
   "wallhitPunish: int=5",
   "wrongDirPunish: int=10",
   "show_plots: bool",
   "startepsilon: float",
   "minepsilon: float",
   "finalepsilonframe: int",
   "epsilon: float")
  ("__init__",
   "policyAction",
   "initForDriving",
   "calculateReward",
   "randomAction",
   "performAction",
   "addToMemory",
   "checkIfAction",
   "canLearn",
   "dauerLearnANN",
   "learnANN",
   "punishLastAction",
   "endEpisode",
   "saveNet",
   "eval_episodeVals",
   "create_QLearnInputs_from_MemoryBatch",
   "freeze-functions");


Class.dqnagent("dqn_rl_agent.Agent")
  ("ff_inputsize: int=30",
   "session: tf.Session",
   "model: DDDQN_model",
   "memory: EfficientMemory")
  ("policyAction",
   "randomAction",
   "preTrain");

Class.blindqnagent("dqn_novision_rl_agent.Agent")
  ("ff_inputsize: int=49",
   "session: tf.Session",
   "usesConv: bool=False",
   "model: DDDQN_model",
   "memory: InefficientMemory")
  ("getAgentState",
   "makeNetUsableOtherInputs",
   "policyAction",
   "randomAction",
   "preTrain");


Class.blinddpgagent("ddpg_novision_rl_agent.Agent")
  ("ff_inputsize: int=30",
   "session: tf.Session",
   "isContinuous: bool=True",
   "usesConv: bool=False",
   "-_noiseState: [float]",
   "model: DDPG_model",
   "memory: InefficientMemory")
  ("makeNetUsableAction",
   "getAgentState",
   "makeNetUsableOtherInputs",
   "make_noisy",
   "randomAction",
   "policyAction",
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
