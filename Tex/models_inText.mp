input metauml;

iClass.iName.iFont.scale := 1.1;
iAssoc.iFont.scale := 1.2;

Interface.model("Model")
  ("__init__()", 
   "initNet(load: bool/string)",
   "save()",
   "inference(oldstates: [s]): [a], _",
   "getAccuracy(batch: [s],[a],[r],[s],[t]>, _): float",
   "statevalue(oldstates: [s]): float",
   "qvalue(oldstates: [s], actions: [a]): float",
   "q_train_step(batch: <[s],[a],[r], [s],[t]>): float");

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
   "q_train_step(batch: <[s],[a],[r],[s],[t]>): float");

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

Class.actornet("actorNet")()();
Class.criticnet("criticNet")()();


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
   "sv_train_step(batch: <[s],[a],[r],[s],[t]>, decay_lr: bool=True): float");

Class.dqn("DuelDQN")()();



beginfig(1)
	model.e = (0,0);
	dpgmod.w = model.e + (40,105);
	dqnmod.nw = dpgmod.sw + (0, -60);
	dqn.w = dqnmod.e + (70,0);
	actor.w = dpgmod.e + (40, 60);
	critic.nw = actor.sw + (0,-10);
	criticnet.n = critic.s + (0,-25);
	actornet.w = actor.e + (30,0);
	drawObjects(model,dpgmod,dqnmod,dqn,actor,critic,actornet,criticnet);
	
	item(iAssoc)("1")(obj.se = actor.w + (-1,-17));
	clink(compositionUni)(dpgmod,actor);

	item(iAssoc)("1")(obj.se = critic.w + (-1,27));
	clink(compositionUni)(dpgmod,critic);

	item(iAssoc)("2")(obj.se = actornet.w + (0,4));
	clink(compositionUni)(actor,actornet);

%	item(iAssoc)("2")(obj.se = criticnet.w + (0,4));
	clink(compositionUni)(critic,criticnet);

	item(iAssoc)("2")(obj.se = dqn.w + (-1,4));
	clink(compositionUni)(dqnmod,dqn);

	clink(realization)(dqnmod, model);
	clink(realization)(dpgmod, model);

endfig;




beginfig(2);
	drawObjects(dqnmod,dqn);

	item(iAssoc)("2")(obj.se = dqn.w + (-1,4));
	clink(compositionUni)(dqnmod,dqn);
endfig;
 
 
beginfig(3);
	drawObjects(dpgmod,actor,critic, actornet, criticnet);

	item(iAssoc)("1")(obj.se = actor.w + (-1,-17));
	clink(compositionUni)(dpgmod,actor);

	item(iAssoc)("1")(obj.se = critic.w + (-1,27));
	clink(compositionUni)(dpgmod,critic);

	item(iAssoc)("2")(obj.se = actornet.w + (0,4));
	clink(compositionUni)(actor,actornet);

	item(iAssoc)("2")(obj.se = criticnet.w + (0,4));
	clink(compositionUni)(critic,criticnet);
endfig;

 
end