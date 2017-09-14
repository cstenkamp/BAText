input metauml;

iClass.iName.iFont.scale := 1.1;
iAssoc.iFont.scale := 1.2;


Class.inemem("InefficientMemory")
  ("capacity: int",
   "-_pointer: int",
   "-_size: int",
   "-_buffer: list")
  ("__init__()",
   "__len__(): int",
   "__getitem__(index: int): <[s],[a],[r],[s],[t]>",
   "append(obj: <[s],[a],[r],[s],[t]>",
   "save_memory()",
   "sample(n: int): [<[s],[a],[r],[s],[t]>]",
   "-_pop()",
   "endEpisode()",
   "punishLastAction(howmuch: float)",
   "average_rewards(fromwhere: int, towhere: int): float");

Class.efficientmem("EfficientMemory")
  ("capacity: int",
   "-_pointer: int",
   "-_size: int",
   "-_ visionvecs: np.array",
   "-_actions: np.array",
   "-_rewards: np.array",
   "-_fEnds: np.array")
  ("__init__()",
   "__len__(): int",
   "__getitem__(index: int): <[s],[a],[r],[s],[t]>",
   "append(obj: <[s],[a],[r],[s],[t]>",
   "save_memory()",
   "sample(n: int): [<[s],[a],[r],[s],[t]>]",
   "-_pop()",
   "endEpisode()",
   "punishLastAction(howmuch: float)");


Interface.mem("Memory")
  ("__init__()",
   "__len__(): int",
   "__getitem__(index: int): <[s],[a],[r],[s],[t]>",
   "append(obj: <[s],[a],[r],[s],[t]>",
   "save_memory()",
   "sample(n: int): [<[s],[a],[r],[s],[t]>]",
   "endEpisode()",
   "punishLastAction(howmuch: float)");

classStereotypes.mem("<<interface>>");

beginfig(1)
	mem.nw = (0,0);
	efficientmem.nw = mem.e + (50,0);
	topToBottom.left(20)(inemem, efficientmem);
	drawObjects(mem, inemem, efficientmem);
	clink(realization)(inemem, mem);
	clink(realization)(efficientmem, mem);
endfig;
