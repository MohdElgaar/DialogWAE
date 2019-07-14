# DialogWAE
Implementation of Xiaodong Gu, Kyunghyun Cho, Jung-Woo Ha, Sunghun Kim, _DialogWAE: Multimodal Response Generation with Conditional Wasserstein Auto-Encoder_, 
Published as a conference paper at ICLR 2019
https://arxiv.org/pdf/1805.12352.pdf

### Data
http://yanran.li/files/ijcnlp_dailydialog.zip

### Embedding Weights
http://nlp.stanford.edu/data/glove.twitter.27B.zip

## Implementation Notes
In the IPython notebook:  
**Version 1** is my implementation of the model and data loading and batching. It achieves very low accuracy; about 0.05 BLEU score on DailyDialog.  
**Version 2** is my implementation of the model integrated with the author's dataloader. It is able to reproduce the paper's results; about 0.3 BLEU score on DailyDialog.

**The difference** between the two dataloaders is that the author's use a type of bucketing. Dialogues of same number of turns are batched together. This makes a big difference in performance.
