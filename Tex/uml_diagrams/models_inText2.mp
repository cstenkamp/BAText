input metauml;

iClass.iName.iFont.scale := 1.1;
iAssoc.iFont.scale := 1.2;

AbstractClass.model("Model")
  ("isPretrain: bool")
  ("__init__()", 
   "initNet(load: bool/string)",
   "save()",
   "inference(oldstates: [s]): [a], _",
   "getAccuracy(batch: [s],[a],[r],[s],[t]>, _): float",
   "statevalue(oldstates: [s]): float",
   "qvalue(oldstates: [s], actions: [a]): float",
   "q_train_step(batch: <[s],[a],[r],[s],[t]>): float",
   "run_inferences(): int",
   "step(): int",
   "pretrain_episode(): int",
   "inc_episode()");

classStereotypes.model("<<interface>>");


Class.dpgmod("DDPG_model")
  ("actor: Actor",
   "critic: Critic",
   "session: tf.Session",
   "isPretrain: bool")
  ("__init__()", 
   "initNet(load: bool/string)",
   "save()",
   "inference(oldstates: [s]): [a], float",
   "getAccuracy(batch: <[s],[a],[r],[s],[t]>, _): float",
   "statevalue(oldstates: [s]): float",
   "qvalue(oldstates: [s], actions: [a]): float",
   "getstatecountfeaturevec(oldstates: [s], actions: [a]): [int]",
   "q_train_step(batch: <[s],[a],[r],[s],[t]>): float",
   "run_inferences(): int",
   "step(): int",
   "pretrain_episode(): int",
   "inc_episode()");

Class.actor("Actor")
  ("session: tf.Session",
   "isPretrain: bool",
   "online: actorNet",
   "target: actorNet")
  ("train(inputs, a_gradient)",
   "predict(inputs, useOnline): [a]",
   "update_target_network()");

Class.critic("Critic")
  ("session: tf.Session",
   "isPretrain: bool",
   "online: criticNet",
   "target: criticNet")
  ("train(inputs, action, target_Q)",
   "predict(inputs, action, useOnline): [float]",
   "action_gradients(inputs, actions): [float]",
   "update_target_network()");



Class.dqnmod("DDDQN_model")
  ("onlineQN: DuelDQN",
   "targetQN: DuelDQN",
   "session: tf.Session",
   "isPretrain: bool")
  ("__init__()", 
   "initNet(load: bool/string)",
   "save()",
   "inference(oldstates: [s]): [a], [float]",
   "getAccuracy(batch: <[s],[a],[r],[s],[t]>, likeDDPG: bool=True): float",
   "statevalue(oldstates: [s]): float",
   "qvalue(oldstates: [s], actions: [a]): float",
   "q_train_step(batch: <[s],[a],[r],[s],[t]>, decay_lr: bool=False): float",
   "sv_train_step(batch: <[s],[a],[r],[s],[t]>, decay_lr: bool=True): float",
   "update_target_network()",
   "run_inferences(): int",
   "step(): int",
   "pretrain_episode(): int",
   "inc_episode()");



beginfig(1)
	model.e = (0,0);
	drawObjects(model);
endfig;


beginfig(2)
	model.e = (0,0);
	dpgmod.w = model.e + (30,105);
	dqnmod.nw = dpgmod.sw + (0, -25);
	actor.w = dpgmod.e + (30, 60);
	critic.nw = actor.sw + (0,-10);
	drawObjects(model,dpgmod,dqnmod,actor,critic);
	
	item(iAssoc)("1")(obj.se = actor.w + (-1,-17));
	clink(compositionUni)(dpgmod,actor);
	
	item(iAssoc)("1")(obj.se = critic.w + (-1,27));
	clink(compositionUni)(dpgmod,critic);
	
	clink(realization)(dqnmod, model);
	clink(realization)(dpgmod, model);

endfig;


 
end