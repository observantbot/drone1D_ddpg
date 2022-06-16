import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curve(x, scores, figure_file):
	running_avg = np.zeros(len(scores))
	for i in range(len(running_avg)):
		running_avg[i] = np.mean(scores[max(0, i-10):(i+1)])
	plt.plot(x, running_avg)
	plt.title('Running average of previous 100 scores')
	plt.savefig(figure_file)


def plotLearning(x, scores, epsilons, filename, lines=None):
	fig=plt.figure()
	ax=fig.add_subplot(111, label="1")
	ax2=fig.add_subplot(111, label="2", frame_on=False)

	ax.plot(x, epsilons, color="C0")
	ax.set_xlabel("time", color="C0")
	ax.set_ylabel("reward2", color="C0")
	ax.tick_params(axis='x', colors="C0")
	ax.tick_params(axis='y', colors="C0")

	N = len(scores)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

	ax2.plot(x, running_avg, color="C1")
	#ax2.xaxis.tick_top()
	ax2.axes.get_xaxis().set_visible(False)
	ax2.yaxis.tick_right()
	#ax2.set_xlabel('x label 2', color="C1")
	ax2.set_ylabel('reward', color="C1")
	#ax2.xaxis.set_label_position('top')
	ax2.yaxis.set_label_position('right')
	#ax2.tick_params(axis='x', colors="C1")
	ax2.tick_params(axis='y', colors="C1")

	if lines is not None:
		for line in lines:
			plt.axvline(x=line)

	plt.savefig(filename)


def evaluate(env, agent, eps_reward_list, eps_time_list, eps_done_list, bestscore):
	eval_eps_rew_list = []
	for j in range(5):
		state = env.reset()
		eval_eps_rew = 0
		for i in range(10000):
			action = agent.get_action(state, training=False)
			ns, r, d, _ = env.step(action[0])
			eval_eps_rew += r
			state = ns
			if d:
				break
		eval_eps_rew_list.append(eval_eps_rew)
	

	if bestscore < np.mean(eval_eps_rew_list):
		bestscore = np.mean(eval_eps_rew_list)
		print('bestscore: ', bestscore)
		agent.save_models()
		dict2 = {'bestscore':[bestscore]}
		df2 = pd.DataFrame(dict2)
		try:
			df3 = pd.read_csv('pybullet/ddpg/bestscore.csv')
			pd.concat([df3, df2]).to_csv('pybullet/ddpg/bestscore.csv', index= False)
		except:
			df2.to_csv('pybullet/ddpg/bestscore.csv', index=False)

	dict = {'eps_reward': eps_reward_list, 
			'time':eps_time_list,
			'done':eps_done_list}
	
	df = pd.DataFrame(dict)
	try:
		df1 = pd.read_csv('pybullet/ddpg/history.csv')
		pd.concat([df1, df]).to_csv('pybullet/ddpg/history.csv', index= False)
	except:
		df.to_csv('pybullet/ddpg/history.csv', index=False)
	
	return bestscore