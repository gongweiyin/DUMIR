from pathlib import Path

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm

MintRec = [
                    'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize',
                    'Agree', 'Taunt', 'Flaunt',
                    'Joke', 'Oppose',
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave',
                    'Prevent', 'Greet', 'Ask for help'
        ]

MintRec2 = [
            'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange',
            'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain',
            'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain',
            'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite',
            'Joke', 'Leave', 'Oppose', 'Plan', 'Praise',
            'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn',
        ]

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """

    cm = confusion_matrix(label_true, label_pred, normalize='true')


    fig, ax = plt.subplots(figsize=(20, 16))

    font_path = Path(__file__).resolve().parents[1] / "Times New Roman.ttf"
    if font_path.exists():
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = "Times New Roman"
    plt.rcParams.update({'font.size': 16})

    im = ax.imshow(cm, cmap='Blues')


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=16)

    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")


    ax.set_xticks(np.arange(len(label_name)))
    ax.set_yticks(np.arange(len(label_name)))
    ax.set_xticklabels(label_name, rotation=80, ha="right", fontsize=16)
    ax.set_yticklabels(label_name, fontsize=16)

    for label in ax.get_xticklabels():
        label.set_fontfamily("Times New Roman")
    for label in ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")


    ax.set_title(
        f"Confusion Matrix of DUMIR on the {title} Dataset",
        fontsize=16,
        fontfamily="Times New Roman"
    )


    for i in range(len(label_name)):
        for j in range(len(label_name)):
            value = cm[j, i] * 100

            if i == j:
                color = "white" if value >= 40 else "black"
            else:
                color = "black"

            ax.text(
                i, j,
                f"{value:.1f}",
                ha="center", va="center",
                color=color, fontsize=16,
                fontfamily="Times New Roman"
            )

    plt.tight_layout()

    if pdf_save_path:
        plt.savefig(pdf_save_path, dpi=dpi, bbox_inches='tight')

    plt.show()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Metrics(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """
    def __init__(self, args):

        self.logger = logging.getLogger(args.logger_name)
        self.eval_metrics = ['acc', 'f1',  'prec', 'rec']

    def __call__(self, y_true, y_pred, show_results = False):

        acc_score = self._acc_score(y_true, y_pred)
        macro_f1 = self._f1_score(y_true, y_pred)
        macro_prec = self._precision_score(y_true, y_pred)
        macro_rec = self._recall_score(y_true, y_pred)

        eval_results = {
            'acc': acc_score,
            'f1': macro_f1,
            'prec': macro_prec,
            'rec': macro_rec,
        }

        if show_results:

            self._show_confusion_matrix(y_true, y_pred)

            self.logger.info("***** Evaluation results *****")
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(round(eval_results[key], 4)))

        return eval_results

    def _acc_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def _f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

    def _precision_score(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='weighted')

    def _recall_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average='macro')

    def _show_confusion_matrix(self, y_true, y_pred):

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=MintRec2, digits=4)

        self.logger.info("%s", report)

        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))

        draw_confusion_matrix(label_true=y_true,
                      label_pred=y_pred,
                      label_name=MintRec2,
                      title="MintRec2",
                      pdf_save_path="2-MIntRec.jpg",
                      dpi=600)