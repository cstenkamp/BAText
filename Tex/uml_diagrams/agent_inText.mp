input metauml;

iClass.iName.iFont.scale := 1.1;
iAssoc.iFont.scale := 1.2;

AbstractClass.agent("AbstractAgent")
  ("isInitialized: bool",
   "action_repeat: int=4",
   "isSupervised: bool=False",
   "isContinuous: bool=False",
   "usesConv: bool=True",
   "usesGUI: bool=False")
  ("getAgentState",
   "initForDriving",
   "performAction",
   "resetUnityAndServer",
   "handle_special_commands");


AbstractClass.rlagent("AbstractRLAgent")
  ("start_fresh: bool",
   "wallhitPunish: int=1",
   "wrongDirPunish: int=5",
   "startepsilon: float",
   "epsilon: float",
   "memory: Inefficientmemory")
  ("policyAction",
   "initForDriving",
   "calculateReward",
   "performAction",
   "randomAction",
   "addToMemory",
   "handle_special_commands",
   "learnANN",
   "dauerLearnANN",
   "punishLastAction",
   "endEpisode",
   "preTrain");


Class.dqnagent("dqn_rl_agent.Agent")
  ("ff_inputsize: int=32",
   "model: DDDQN_model",
   "usesGUI: bool=True",
   "memory: EfficientMemory")
  ("initForDriving",
   "policyAction"); 

Class.blindqnagent("dqn_novision_rl_agent.Agent")
  ("ff_inputsize: int=156",
   "usesConv: bool=False",
   "model: DDDQN_model",
   "usesGUI: bool=True",)
  ("getAgentState",
   "policyAction");

Class.ddpgagent("ddpg_rl_agent.Agent")
  ("ff_inputsize: int=32",
   "model: DDPG_model",
   "usesGUI: bool=True",
   "isContinous: bool=True",
   "memory: EfficientMemory")
  ("initForDriving",
   "-make_noisy",
   "randomAction",
   "policyAction",
   "preTrain"); 


Class.blinddpgagent("ddpg_novision_rl_agent.Agent")
  ("ff_inputsize: int=156",
   "model: DDPG_model",
   "usesGUI: bool=True",
   "usesConv: bool=False",
   "isContinous: bool=True")
  ("getAgentState",
   "-make_noisy",
   "randomAction",
   "policyAction",
   "preTrain"); 
   
   
Class.svagent("dqn_sv_agent.Agent")
  ("ff_inputsize: int=32",
   "model: DDDQN_model",
   "isSupervised: bool=True")
  ("policyAction",
   "preTrain");    


beginfig(1)

	agent.e = (0,0);
	rlagent.w = agent.e + (50, 0);
	dqnagent.nw = rlagent.e + (50,200);	
	blindqnagent.nw = rlagent.e + (50,110);	
	ddpgagent.nw = rlagent.e + (50, 20);	
	blinddpgagent.nw = rlagent.e + (50,-112);	
	svagent.n = agent.s + (0, -60);
	
	drawObjects(agent, rlagent, dqnagent, blindqnagent, ddpgagent, blinddpgagent, svagent);


	link(inheritance)(rlagent.w -- agent.e);
	link(inheritance)(svagent.n -- agent.s);
	
	link(inheritance)(pathStepX(dqnagent.w, rlagent.e, -15));
	link(inheritance)(pathStepX(blindqnagent.w, rlagent.e, -15));
	link(inheritance)(pathStepX(blinddpgagent.w, rlagent.e, -15));
	link(inheritance)(pathStepX(ddpgagent.w, rlagent.e, -15));


endfig;


end
