import numpy as np
try:
    import _pickle as pickle
except:
    import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os.path as osp
sns.set_context("paper", font_scale=1.5)
flatui = ["#0a4d8c", "#ba0c2f", "#49a942", "#84a40b", "#004851", "#0095c8", "#4d2177", "#7c3a2d"]

from matplotlib import rc
from matplotlib import rcParams
rc('text', usetex=True)
rcParams['backend'] = 'ps'
rcParams['text.latex.preamble'] = ["\\usepackage{gensymb}"]
rcParams['font.size'] = 12
rcParams['legend.fontsize'] = 12
rc('font', **{'family':'serif', 'serif':['Computer Modern'],
       'monospace': ['Computer Modern Typewriter']})

STAGES = 4

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path",
                        help="root path of results", type=str,
                        default="./results/DaRe")
    parser.add_argument("--figname",
                        help="feature name", type=str,
                        default="budgeted_stream")
    return parser.parse_args()

# flops (unit: macc) are calculated using http://dgschwend.github.io/netscopes
SVDNet_R_RE = [3888220399, 87.1]
SVDNet_C = [709398816, 80.5]
TriNet_R = [2528948224, 84.92]
IDE_R_KISSME = [3886925312, 73.60]
IDE_C_KISSME = [674923280, 58.61]

def main(args):
    distance_confidence_info = pickle.load(open(osp.join(args.result_path, "distance_confidence_info.pkl"), "rb"))
    margin_confidence_info = pickle.load(open(osp.join(args.result_path, "margin_confidence_info.pkl"), "rb"))
    random_info = pickle.load(open(osp.join(args.result_path, "random_info.pkl"), "rb"))
    distance_confidence_info['CMCs'] = [cmc * 100 for cmc in distance_confidence_info['CMCs']]
    margin_confidence_info['CMCs'] = [cmc * 100 for cmc in margin_confidence_info['CMCs']]
    random_info['CMCs'] = [cmc * 100 for cmc in random_info['CMCs']]

    with sns.axes_style("white"):
        fig = plt.figure(figsize=(6, 4.5))
        ax  = fig.add_subplot(111)
        ax.plot(random_info['resulted_budgets'], random_info['CMCs'], marker='.', linewidth=2.5, markersize=0, label="DaRe(R)+RE (random)", color=flatui[0])
        ax.plot(distance_confidence_info['resulted_budgets'], distance_confidence_info['CMCs'], marker='*', linewidth=2.5, markersize=0, label="DaRe(R)+RE (distance)", color=flatui[1])
        ax.plot(margin_confidence_info['resulted_budgets'], margin_confidence_info['CMCs'], marker='*', linewidth=2.5, markersize=0, label="DaRe(R)+RE (margin)", color=flatui[2])

        ax.scatter(SVDNet_R_RE[0], SVDNet_R_RE[1], marker='*', s=150, label="SVDNet(R)+RE", color=flatui[3])
        ax.scatter(IDE_R_KISSME[0], IDE_R_KISSME[1], marker='h', s=100, label="IDE(R)+KISSME", color=flatui[4])
        ax.scatter(IDE_C_KISSME[0], IDE_C_KISSME[1], marker='o', s=100, label="IDE(C)+KISSME", color=flatui[5])
        ax.scatter(TriNet_R[0], TriNet_R[1], marker='D', s=60, label="TriNet(R)", color=flatui[6])
        ax.scatter(SVDNet_C[0], SVDNet_C[1], marker='p', s=100, label="SVDNet(C)", color=flatui[7])
        plt.xlabel("Average Budget (in MUL-ADD)", size=15)
        plt.ylabel("CMC Rank 1 Accuracy (\%)", size=15)
        handles, labels = ax.get_legend_handles_labels()
        label_order = ['TriNet(R)', 'SVDNet(C)', 'SVDNet(R)+RE', 'IDE(R)+KISSME', 'IDE(C)+KISSME', 'DaRe(R)+RE (random)', 'DaRe(R)+RE (distance)', 'DaRe(R)+RE (margin)']
        new_handles = []
        for l in label_order:
            for i in range(len(labels)):
                if labels[i] == l:
                    new_handles.append(handles[i])
        ax.legend(new_handles, label_order, loc='lower right')
        plt.grid(linestyle='dotted')
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.xlim(3e8, 4.5e9)
        plt.ylim(55, 95)
        plt.savefig(args.figname + ".pdf", bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main(parse_args())