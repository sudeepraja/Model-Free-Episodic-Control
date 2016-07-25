# Model-Free-Episodic-Control
Implementation of the Model Free Episodic Control paper by Deep Mind : http://arxiv.org/abs/1606.04460

#Introduction
This is an implementation of Episodic Control Agent. The implementation is a modification of [ShibiHe/Model-Free-Episodic-Control
](https://github.com/ShibiHe/Model-Free-Episodic-Control).

#Dependencies

Game roms should be stored in directory *roms* which stays next to this folder.

Parent Folder

├ Model-Free-Episodic-Control -> source codes + README.md

└ roms -> game roms

###Dependencies for running Episodic Control

[Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) or [OpenAI gym](https://github.com/openai/gym)
 
 Numpy and SciPy
 
 A reasonable CPU
 
# Running
examples:

`python run_episodic_control.py`

To get more running details, we can use `python run_episodic_control.py -h`
