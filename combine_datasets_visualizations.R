library(dplyr)
library(ggplot2)
library(reshape2)

### read all datasets and combine human annotator data sets
gpt4o_res_comb_df = read.csv(file='data/openAI_res_logprobs_variance_df.csv', sep = ",")
gpt4o_res_comb_df$logprobs_mean = gpt4o_res_comb_df$logprobs/length(gpt4o_res_comb_df$logprobs)
gpt4o_res_comb_df$maj_vote <- ifelse(gpt4o_res_comb_df$maj_vote == "w", "f", gpt4o_res_comb_df$maj_vote)
# View(gpt4o_res_comb_df)

ChatAI_70B_res_comb_df = read.csv(file='data/meta-llama-3.1-70b-instruct_ChatAI_res_logprobs_variance_df.csv', sep = ",")
ChatAI_70B_res_comb_df$logprobs_mean = ChatAI_70B_res_comb_df$logprobs/length(ChatAI_70B_res_comb_df$logprobs)
ChatAI_70B_res_comb_df$maj_vote <- ifelse(ChatAI_70B_res_comb_df$maj_vote == "w", "f", ChatAI_70B_res_comb_df$maj_vote)
# View(ChatAI_70B_res_comb_df)

ChatAI_8B_res_comb_df = read.csv(file='data/meta-llama-3.1-8b-instruct_ChatAI_res_logprobs_variance_df.csv', sep = ",")
ChatAI_8B_res_comb_df$logprobs_mean = ChatAI_8B_res_comb_df$logprobs/length(ChatAI_8B_res_comb_df$logprobs)
ChatAI_8B_res_comb_df$maj_vote <- ifelse(ChatAI_8B_res_comb_df$maj_vote == "w", "f", ChatAI_8B_res_comb_df$maj_vote)
# View(ChatAI_8B_res_comb_df)

prolific_distribution_wide = read.csv(file='data/aggregated_human_annotators.csv', sep = ",")

comb <- left_join(prolific_distribution_wide[,1:6], gpt4o_res_comb_df[,c(1,3,5,7,8,9)], by='np', copy = FALSE, suffix = c("_prolific", "_gpt4o"), keep = NULL) %>%
  left_join(., ChatAI_70B_res_comb_df[,c(1,3,5,7,8,9)], by='np', suffix = c("","_llama_3.1_70B")) %>%
  left_join(., ChatAI_8B_res_comb_df[,c(1,3,5,7,8,9)], by='np', suffix = c("","_llama_3.1_8B")) 
comb = comb[,c(1,2,4,3,6,5,9,10,11,7,8,14,15,16,12,13,19,20,21,17,18)]  
names(comb)[12:16] = c("m_llama_3.1_70B", "w_llama_3.1_70B", "n_llama_3.1_70B", "maj_vote_llama_3.1_70B", "agreement_llama_3.1_70B")
#View(comb)

write.csv(comb, "results/aggregated_combination_llm_human_annotators.csv")



########################################################
################## Confusion Matrices ##################
########################################################

library(caret)
library(Metrics)

rem_target = comb[which(comb$maj_vote_prolific=="unklar"),]$np
comb_tmp = comb[-which(comb$np == rem_target),]

## Prolific v GPT-4o
maj_vote_gpt4o <- as.factor(comb_tmp$maj_vote_gpt4o)
maj_vote_gpt4o <- factor(maj_vote_gpt4o, levels = c("m", "n", "f"))

maj_vote_prolific <- as.factor(comb_tmp$maj_vote_prolific)
maj_vote_prolific <- factor(maj_vote_prolific, levels = c("m", "n", "f"))

prolific_gpt4o_cm = confusionMatrix(data = maj_vote_gpt4o, reference = maj_vote_prolific, dnn = c("GPT4o", "Human_Annotators"), positive = "True")
prolific_gpt4o_plt <- as.data.frame(prolific_gpt4o_cm$table)

tmp = aggregate(prolific_gpt4o_plt$Freq, by=list(Category=prolific_gpt4o_plt$Human_Annotators), FUN=sum)
names(tmp) = c("GPT4o", "Sum")
prolific_gpt4o_plt_comb <- left_join(prolific_gpt4o_plt, tmp, by='GPT4o', copy = FALSE, keep = NULL)
prolific_gpt4o_plt_comb$prop = (prolific_gpt4o_plt_comb$Freq/prolific_gpt4o_plt_comb$Sum)*100
prolific_gpt4o_plt_comb$GPT4o <- factor(prolific_gpt4o_plt_comb$GPT4o, levels=rev(levels(prolific_gpt4o_plt_comb$GPT4o)))

prolific_gpt4o_cm_vis = ggplot(prolific_gpt4o_plt_comb, aes(Human_Annotators,GPT4o, fill= prop)) +
  geom_tile() + 
  geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  scale_x_discrete(labels=c("m","n","f")) +
  scale_y_discrete(labels=c("f","n","m")) +
  theme(legend.position="none", axis.title.x=element_blank()) +
  ylab("GPT-4o")
prolific_gpt4o_cm_vis

## Prolific v Llama 3.1 70B
maj_vote_llama_3.1_70B <- as.factor(comb_tmp$maj_vote_llama_3.1_70B)
maj_vote_llama_3.1_70B <- factor(maj_vote_llama_3.1_70B, levels = c("m", "n", "f"))

maj_vote_prolific <- as.factor(comb_tmp$maj_vote_prolific)
maj_vote_prolific <- factor(maj_vote_prolific, levels = c("m", "n", "f"))

prolific_llama_3.1_70B_cm = confusionMatrix(data = maj_vote_llama_3.1_70B, reference = maj_vote_prolific, dnn = c("LLaMA_3.1_70B", "Human_Annotators"), positive = "True")
prolific_llama_3.1_70B_plt <- as.data.frame(prolific_llama_3.1_70B_cm$table)

tmp = aggregate(prolific_llama_3.1_70B_plt$Freq, by=list(Category=prolific_llama_3.1_70B_plt$Human_Annotators), FUN=sum)
names(tmp) = c("LLaMA_3.1_70B", "Sum")
prolific_llama_3.1_70B_plt_comb <- left_join(prolific_llama_3.1_70B_plt, tmp, by='LLaMA_3.1_70B', copy = FALSE, keep = NULL)
prolific_llama_3.1_70B_plt_comb$prop = (prolific_llama_3.1_70B_plt_comb$Freq/prolific_llama_3.1_70B_plt_comb$Sum)*100
prolific_llama_3.1_70B_plt_comb$LLaMA_3.1_70B <- factor(prolific_llama_3.1_70B_plt_comb$LLaMA_3.1_70B, levels=rev(levels(prolific_llama_3.1_70B_plt_comb$LLaMA_3.1_70B)))

prolific_llama_3.1_70B_cm_vis = ggplot(prolific_llama_3.1_70B_plt_comb, aes(Human_Annotators,LLaMA_3.1_70B, fill= prop)) +
  geom_tile() + 
  geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  scale_x_discrete(labels=c("m","n","f")) +
  scale_y_discrete(labels=c("f","n","m")) +
  theme(legend.position="none", axis.title.x=element_blank())
  # theme(legend.position="none")
prolific_llama_3.1_70B_cm_vis


## Prolific v Llama 3.1 8B

maj_vote_llama_3.1_8B <- as.factor(comb_tmp$maj_vote_llama_3.1_8B)
maj_vote_llama_3.1_8B <- factor(maj_vote_llama_3.1_8B, levels = c("m", "n", "f"))

maj_vote_prolific <- as.factor(comb_tmp$maj_vote_prolific)
maj_vote_prolific <- factor(maj_vote_prolific, levels = c("m", "n", "f"))

prolific_llama_3.1_8B_cm = confusionMatrix(data = maj_vote_llama_3.1_8B, reference = maj_vote_prolific, dnn = c("LLaMA_3.1_8B", "Human_Annotators"), positive = "True")
prolific_llama_3.1_8B_plt <- as.data.frame(prolific_llama_3.1_8B_cm$table)

tmp = aggregate(prolific_llama_3.1_8B_plt$Freq, by=list(Category=prolific_llama_3.1_8B_plt$Human_Annotators), FUN=sum)
names(tmp) = c("LLaMA_3.1_8B", "Sum")
prolific_llama_3.1_8B_plt_comb <- left_join(prolific_llama_3.1_8B_plt, tmp, by='LLaMA_3.1_8B', copy = FALSE, keep = NULL)
prolific_llama_3.1_8B_plt_comb$prop = (prolific_llama_3.1_8B_plt_comb$Freq/prolific_llama_3.1_8B_plt_comb$Sum)*100
prolific_llama_3.1_8B_plt_comb$LLaMA_3.1_8B <- factor(prolific_llama_3.1_8B_plt_comb$LLaMA_3.1_8B, levels=rev(levels(prolific_llama_3.1_8B_plt_comb$LLaMA_3.1_8B)))

prolific_llama_3.1_8B_cm_vis = ggplot(prolific_llama_3.1_8B_plt_comb, aes(Human_Annotators,LLaMA_3.1_8B, fill= prop)) +
  geom_tile() + 
  geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  scale_x_discrete(labels=c("m","n","f")) +
  scale_y_discrete(labels=c("f","n","m")) +
  theme(legend.position="none")
  # theme(legend.position="none", axis.title.x=element_blank())
prolific_llama_3.1_8B_cm_vis

#### Combine single plots into one object
library(ggpubr)
fig_comb_confusion = ggarrange(prolific_gpt4o_cm_vis, prolific_llama_3.1_70B_cm_vis, prolific_llama_3.1_8B_cm_vis, 
          ncol = 1, nrow = 3)
fig_comb_confusion
ggsave("results/fig_comb_confusion.png", fig_comb_confusion, bg='transparent', width = 15, height = 10, units = "cm", dpi = 300)

#### Precision, Recall and F1

### GPT4o
library(caret)
library(Metrics)

# Create predicted and actual class labels
gpt4o_prec = prolific_gpt4o_cm$byClass[,"Precision"]
gpt4o_recall = prolific_gpt4o_cm$byClass[,"Recall"]
gpt4o_f1 = prolific_gpt4o_cm$byClass[,"F1"]

llama70B_prec = prolific_llama_3.1_70B_cm$byClass[,"Precision"]
llama70B_recall = prolific_llama_3.1_70B_cm$byClass[,"Recall"]
llama70B_f1 = prolific_llama_3.1_70B_cm$byClass[,"F1"]

llama8B_prec = prolific_llama_3.1_8B_cm$byClass[,"Precision"]
llama8B_recall = prolific_llama_3.1_8B_cm$byClass[,"Recall"]
llama8B_f1 = prolific_llama_3.1_8B_cm$byClass[,"F1"]

comb_prec_recall_f1_macro = data.frame(Class=c("m", "n", "w"), GPT4o_Precision = round(gpt4o_prec,3), GPT4o_Recall = round(gpt4o_recall,3), GPT4o_F1 = round(gpt4o_f1,3),
                                       LLama_3.1_70B_Precision = round(llama70B_prec, 3), LLama_3.1_70B_Recall = round(llama70B_recall, 3), LLama_3.1_70B_F1 = round(llama70B_f1,3),
                                       LLama_3.1_8B_Precision = round(llama8B_prec,3), LLama_3.1_8B_Recall = round(llama8B_recall,3), LLama_3.1_8B_F1 = round(llama8B_f1,3))
comb_prec_recall_f1_macro

library(ggplot2)
library(patchwork)
library(gridExtra) # for tableGrob
library(dplyr)# to help with creating a minimal table
library(tibble) # to remove rownames
library(gtable)

padding <- unit(2,"mm")
gpt4o_prf1 = comb_prec_recall_f1_macro[,c(1,2,3,4)]
names(gpt4o_prf1) <- c("Class", "Precision", "Recall", "F1")
tbl1 <- tableGrob(gpt4o_prf1, theme=ttheme_minimal(base_size = 9), rows=NULL)
title_tbl1 <- textGrob("GPT-4o",gp=gpar(fontsize=10))
tbl1 <- gtable_add_rows(
  tbl1, 
  heights = grobHeight(title_tbl1) + padding,
  pos = 0)
tbl1 <- gtable_add_grob(
  tbl1, 
  title_tbl1, 
  1, 1, 1, ncol(tbl1))

llama70B_prf1 = comb_prec_recall_f1_macro[,c(1,5,6,7)]
names(llama70B_prf1) <- c("Class", "Precision", "Recall", "F1")
tbl2 <- tableGrob(llama70B_prf1, theme=ttheme_minimal(base_size = 9), rows=NULL)
title_tbl2 <- textGrob("LLaMA 3.1 70B",gp=gpar(fontsize=10))
tbl2 <- gtable_add_rows(
  tbl2, 
  heights = grobHeight(title_tbl2) + padding,
  pos = 0)
tbl2 <- gtable_add_grob(
  tbl2, 
  title_tbl2, 
  1, 1, 1, ncol(tbl2))

llama8B_prf1 = comb_prec_recall_f1_macro[,c(1,8,9,10)]
names(llama8B_prf1) <- c("Class", "Precision", "Recall", "F1")
tbl3 <- tableGrob(llama8B_prf1, theme=ttheme_minimal(base_size = 9), rows=NULL)
title_tbl3 <- textGrob("LLaMA 3.1 8B",gp=gpar(fontsize=10))
tbl3 <- gtable_add_rows(
  tbl3, 
  heights = grobHeight(title_tbl3) + padding,
  pos = 0)
tbl3 <- gtable_add_grob(
  tbl3, 
  title_tbl3, 
  1, 1, 1, ncol(tbl3))


fig_comb_confusion_prf1 = ggarrange(prolific_gpt4o_cm_vis, tbl1, prolific_llama_3.1_70B_cm_vis, tbl2, prolific_llama_3.1_8B_cm_vis, tbl3,
          #labels = c("A", "B", "C"),
          ncol = 2, nrow = 3)
fig_comb_confusion_prf1

ggsave("results/fig_comb_confusion_prf1.png", fig_comb_confusion_prf1, bg='transparent', width = 15, height = 10, units = "cm", dpi = 300)

### Calculate Random Baseline
set.seed(42)
probs_m_n_f = as.vector(table(maj_vote_prolific))
probs_m_n_f = probs_m_n_f/115
sum(probs_m_n_f)
labels_sampling = c("m", "n", "f")
random_baseline <- sample(labels_sampling,115,replace=TRUE,prob=probs_m_n_f)
random_baseline = factor(random_baseline, levels = c("m", "n", "f"))
prolific_random_cm = confusionMatrix(data = random_baseline, reference = maj_vote_prolific, dnn = c("Random_Baseline", "Human_Annotators"), positive = "True")
prolific_random_plt <- as.data.frame(prolific_random_cm$table)

random_prec = prolific_random_cm$byClass[,"Precision"]
random_recall = prolific_random_cm$byClass[,"Recall"]
random_f1 = prolific_random_cm$byClass[,"F1"]

table(random_baseline)

########################################################
###################### Agreement  ######################
########################################################

### combine all agreement visualizations  into one

library(viridis)
library(ggrepel)

prolific_cutoff = 1
prolific_distribution_wide$label_scatter <- ifelse(prolific_distribution_wide$agreement == prolific_cutoff, prolific_distribution_wide$np, '')
prolific_distribution_wide$maj_vote = ifelse(prolific_distribution_wide$maj_vote=="unklar", "unclear", prolific_distribution_wide$maj_vote)
prolific_distribution_wide$maj_vote <- as.factor(prolific_distribution_wide$maj_vote)
prolific_distribution_wide$maj_vote <- factor(prolific_distribution_wide$maj_vote, levels = c("m", "n", "f", "unclear"))

prolific_agree_item = ggplot(prolific_distribution_wide, aes(y=agreement, x=maj_vote)) +
  geom_jitter(width = 0.1, height = 0.0) +
  geom_text_repel(aes(label=label_scatter), max.overlaps = 20) +
  labs(
       # subtitle = "Human Annotators; labels for agreement = 1",
       x = "Majority Vote",
       y = "Agreement per Item") +
  theme(
    axis.title.y = element_text(color = "black", size = 14),
    axis.text.y = element_text(angle = 0, size = 14),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 14),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="none"
  ) +
  theme_light()
prolific_agree_item


openAI_cutoff = 1
gpt4o_res_comb_df$label_scatter <- ifelse(gpt4o_res_comb_df$agreement < openAI_cutoff, gpt4o_res_comb_df$np, '')
gpt4o_res_comb_df$maj_vote <- as.factor(gpt4o_res_comb_df$maj_vote)
gpt4o_res_comb_df$maj_vote <- factor(gpt4o_res_comb_df$maj_vote, levels = c("m", "n", "f"))

openAI_agree_item = ggplot(gpt4o_res_comb_df, aes(y=agreement, x=maj_vote)) + 
  geom_jitter(width = 0.1, height = 0.0) +
  geom_text_repel(aes(label=label_scatter), max.overlaps = 20) +
  labs(
    # subtitle = paste0("GPT-4o; labels for agreement < ", openAI_cutoff),
       x = "Majority Vote",
       y = "Agreement per Item") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 14),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="none"
  ) +
  theme_light()
openAI_agree_item


ChatAI_70B_cutoff = 1
ChatAI_70B_res_comb_df$label_scatter <- ifelse(ChatAI_70B_res_comb_df$agreement < ChatAI_70B_cutoff, ChatAI_70B_res_comb_df$np, '')
ChatAI_70B_res_comb_df$maj_vote <- as.factor(ChatAI_70B_res_comb_df$maj_vote)
ChatAI_70B_res_comb_df$maj_vote <- factor(ChatAI_70B_res_comb_df$maj_vote, levels = c("m", "n", "f"))

ChatAI_70B_agree_item = ggplot(ChatAI_70B_res_comb_df, aes(y=agreement, x=maj_vote)) + 
  geom_jitter(width = 0.1, height = 0.0) +
  geom_text_repel(aes(label=label_scatter), max.overlaps = 20) +
  labs(#title = "LLaMA 3.1 70B - Agreement per Item",
       # subtitle = paste0("LLaMA 3.1 70B; labels for agreement < ", ChatAI_70B_cutoff),
       x = "Majority Vote",
       y = "Agreement per Item") +
  theme(
    axis.title.y = element_text(color = "black", size = 14),
    axis.text.y = element_text(angle = 0, size = 14),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 14),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="none"
  ) +
  theme_light()
ChatAI_70B_agree_item

ChatAI_8B_cutoff = 1
ChatAI_8B_res_comb_df$label_scatter <- ifelse(ChatAI_8B_res_comb_df$agreement < ChatAI_8B_cutoff, ChatAI_8B_res_comb_df$np, '')
ChatAI_8B_res_comb_df$label_scatter <- ifelse(ChatAI_8B_res_comb_df$agreement < ChatAI_8B_cutoff, ChatAI_8B_res_comb_df$np, '')
ChatAI_8B_res_comb_df$maj_vote <- as.factor(ChatAI_8B_res_comb_df$maj_vote)
ChatAI_8B_res_comb_df$maj_vote <- factor(ChatAI_8B_res_comb_df$maj_vote, levels = c("m", "n", "f"))


ChatAI_8B_agree_item = ggplot(ChatAI_8B_res_comb_df, aes(y=agreement, x=maj_vote)) + 
  geom_jitter(width = 0.1, height = 0.0) +
  geom_text_repel(aes(label=label_scatter), max.overlaps = 20) +
  labs(#title = "LLaMA 3.1 8B - Agreement per Item",
       # subtitle = paste0("LLaMA 3.1 8B; labels for agreement < ", ChatAI_8B_cutoff),
       x = "Majority Vote",
       y = "Agreement per Item") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 14),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="none"
  ) +
  theme_light()
ChatAI_8B_agree_item

library(ggpubr)
library(grid)

fig_comb_agreement_item = ggarrange(prolific_agree_item +
                  theme(axis.title.x = element_blank()), 
          openAI_agree_item + 
            theme(axis.text.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  axis.title.y = element_blank(), 
                  axis.title.x = element_blank()),
          ChatAI_70B_agree_item + 
                  theme(axis.title.x = element_blank()),
          ChatAI_8B_agree_item + 
            theme(axis.text.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  axis.title.y = element_blank(),
                  axis.title.x = element_blank()),
          nrow = 2,
          ncol = 2)

title <- expression(atop(bold("Agreement per Item")))
fig_comb_agreement_item = annotate_figure(fig_comb_agreement_item,
                top = textGrob(title, gp = gpar(fontsize = 14)), bottom = textGrob("Majority Vote", gp = gpar(fontsize = 14)))
fig_comb_agreement_item
ggsave("results/fig_comb_agreement_item.png", fig_comb_agreement_item, bg='transparent', width = 20, height = 15, units = "cm", dpi = 300)



#### Sub 1 agreement LLM, position in Human Annotators
sub_1_agreement = unique(c(ChatAI_70B_res_comb_df$label_scatter, ChatAI_8B_res_comb_df$label_scatter, gpt4o_res_comb_df$label_scatter))
sub_1_agreement = sub_1_agreement[2:length(sub_1_agreement)]

prolific_distribution_wide_llm_sub_1 = prolific_distribution_wide
prolific_distribution_wide_llm_sub_1$label_scatter <- ifelse(prolific_distribution_wide$np %in% sub_1_agreement, prolific_distribution_wide$np, '')

prolific_agree_item_llm_sub_1 = ggplot(prolific_distribution_wide_llm_sub_1, aes(y=agreement, x=maj_vote)) +
  geom_jitter(width = 0.1, height = 0.0) +
  geom_text_repel(aes(label=label_scatter), max.overlaps = 20) +
  # scale_color_viridis() +
  labs(
    subtitle = "Human Annotators; labels items with LLM agreement < 1",
    # subtitle = paste0(length(prolific_distribution_wide$le_text), " items, ", prolific_part_per_item, " participants/item;", " labels for agreement < ", prolific_cutoff),
    x = "Majority Vote",
    y = "Agreement per Item") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 12),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="none"
  )
prolific_agree_item_llm_sub_1

ggsave("results/fig_agreement_llm_sub_1.png", prolific_agree_item_llm_sub_1, bg='transparent', width = 20, height = 20, units = "cm", dpi = 300)


########################################################
################### circular barplot  ##################
########################################################


# library
library(tidyverse)
library(viridis)


gpt4o_sub1 = gpt4o_res_comb_df[which(gpt4o_res_comb_df$agreement<1),]
gpt4o_sub1 = gpt4o_sub1[,c(1,7,8,9)]
gpt4o_sub1$model = "GPT-4o"

llama70b_sub1 = ChatAI_70B_res_comb_df[which(ChatAI_70B_res_comb_df$agreement<1),]
llama70b_sub1 = llama70b_sub1[,c(1,7,8,9)]
llama70b_sub1$model = "LLaMA 3.1 70B"

llama8b_sub1 = ChatAI_8B_res_comb_df[which(ChatAI_8B_res_comb_df$agreement<1),]
llama8b_sub1 = llama8b_sub1[,c(1,7,8,9)]
llama8b_sub1$model = "LLaMA 3.1 8B"

names(gpt4o_sub1)
names(llama70b_sub1)
names(llama8b_sub1)

sub_1_agreement_new = rbind(gpt4o_sub1, llama70b_sub1, llama8b_sub1)
sub_1_agreement_new = sub_1_agreement_new[,c(1,5,2,4,3)]
names(sub_1_agreement_new) <-  c("individual", "group", "value1", "value2", "value3")
sub_1_agreement_new$group = as.factor(sub_1_agreement_new$group)


# Transform data in a tidy format (long format)
data = sub_1_agreement_new
data <- data %>% gather(key = "observation", value="value", -c(1,2))

# Set a number of 'empty bar' to add at the end of each group
empty_bar <- 2
nObsType <- nlevels(as.factor(data$observation))
to_add <- data.frame(matrix(NA, empty_bar*nlevels(as.factor(data$group))*nObsType, ncol(data)) )
colnames(to_add) <- colnames(data)
to_add$group <- rep(levels(data$group), each=empty_bar*nObsType )
data <- rbind(data, to_add)
data <- data %>% arrange(group, individual)
data$id <- rep( seq(1, nrow(data)/nObsType) , each=nObsType)

# Get the name and the y position of each label
label_data <- data %>% dplyr::group_by(id, individual) %>% dplyr::summarize(tot=sum(value))
number_of_bar <- nrow(label_data)
angle <- 90 - 360 * (label_data$id-0.5) /number_of_bar     # I substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
label_data$hjust <- ifelse( angle < -90, 1, 0)
label_data$angle <- ifelse(angle < -90, angle+180, angle)

# prepare a data frame for base lines
base_data <- data %>% 
  group_by(group) %>% 
  dplyr::summarize(start=min(id), end=max(id) - empty_bar) %>% 
  rowwise() %>% 
  dplyr::mutate(title=mean(c(start, end)))

# prepare a data frame for grid (scales)
grid_data <- base_data
grid_data$end <- grid_data$end[ c( nrow(grid_data), 1:nrow(grid_data)-1)] + 1
grid_data$start <- grid_data$start - 1
grid_data <- grid_data[-1,]

label_data$individual <-  ifelse(label_data$individual == "Leichtgewicht", "Leicht-\ngewicht", label_data$individual)
label_data$individual <-  ifelse(label_data$individual == "Jahrhunderttalent", "Jahrhundert-\ntalent", label_data$individual)
label_data$individual <-  ifelse(label_data$individual == "Schnapsdrossel", "Schnaps-\ndrossel", label_data$individual)


# Make the plot
llm_sub1_p <- ggplot(data) +      
  
  # Add the stacked bar
  geom_bar(aes(x=as.factor(id), y=value, fill=observation), stat="identity", alpha=0.5)+
  # scale_y_continuous(expand = expansion(add = c(10,100))) +
  scale_fill_viridis(discrete=TRUE, labels = c("m", "n", "f"), na.translate = F) +

  # Add a val=100/75/50/25 lines. I do it at the beginning to make sur barplots are OVER it.
  geom_segment(data=grid_data, aes(x = end, y = 0, xend = start, yend = 0), colour = "grey", alpha=1, linewidth=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 10, xend = start, yend = 10), colour = "grey", alpha=1, linewidth=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 20, xend = start, yend = 20), colour = "grey", alpha=1, linewidth=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 33, xend = start, yend = 33), colour = "grey", alpha=1, linewidth=0.3 , inherit.aes = FALSE ) +

  # Add text showing the value of each 100/75/50/25 lines
  ggplot2::annotate("text", x = rep(max(data$id),4), y = c(0, 10, 20, 33), label = c("0", "10", "20", "33") , color="grey", size=3 , angle=0, fontface="bold", hjust=1) + ylim(-50,max(label_data$tot, na.rm=T)) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(rep(-0,4), "cm"),
    # plot.margin = unit(c(0,0,0,2), "cm"),
  ) +
  guides(fill = guide_legend(title = "Label")) + 
  coord_polar() +
  
  # Add labels on top of each bar
  geom_text(data=label_data, aes(x=id, y=tot-5, label=individual, hjust=hjust), color="black", fontface="bold",alpha=0.6, size=4, angle= label_data$angle, inherit.aes = FALSE) +
  # Add base line information
  geom_segment(data=base_data, aes(x = start, y = -5, xend = end, yend = -2), colour = "black", alpha=0.8, size=0.6 , inherit.aes = FALSE )  +
  geom_text(data=base_data, aes(x = title, y = -18, label=group), hjust=c(1,1,0), colour = "black", alpha=0.8, size=4, fontface="bold", inherit.aes = FALSE)

llm_sub1_p
ggsave("results/fig_circular_barplot_llmsub1.png", llm_sub1_p, bg='transparent', width = 20, height = 20, units = "cm", dpi = 300)

########## Human Annotators

prolific_sub_1_agreement = prolific_distribution_wide[which(prolific_distribution_wide$agreement<1),]
prolific_sub_1_agreement = prolific_sub_1_agreement[,c(1,5,2,3,4)]
names(prolific_sub_1_agreement) <-  c("individual", "group", "value1", "value2", "value3")
prolific_sub_1_agreement$group = as.factor(prolific_sub_1_agreement$group)

# Transform data in a tidy format (long format)
data = prolific_sub_1_agreement
data <- data %>% gather(key = "observation", value="value", -c(1,2))

# Set a number of 'empty bar' to add at the end of each group
empty_bar <- 2
nObsType <- nlevels(as.factor(data$observation))
to_add <- data.frame(matrix(NA, empty_bar*nlevels(as.factor(data$group))*nObsType, ncol(data)) )
colnames(to_add) <- colnames(data)
to_add$group <- rep(levels(data$group), each=empty_bar*nObsType )
data <- rbind(data, to_add)
data <- data %>% arrange(group, individual)
data$id <- rep( seq(1, nrow(data)/nObsType) , each=nObsType)

# Get the name and the y position of each label
label_data <- data %>% dplyr::group_by(id, individual) %>% dplyr::summarize(tot=sum(value))
number_of_bar <- nrow(label_data)
angle <- 90 - 360 * (label_data$id-0.5) /number_of_bar     # I substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
label_data$hjust <- ifelse( angle < -90, 1, 0)
label_data$angle <- ifelse(angle < -90, angle+180, angle)

# prepare a data frame for base lines
base_data <- data %>% 
  group_by(group) %>% 
  dplyr::summarize(start=min(id), end=max(id) - empty_bar) %>% 
  rowwise() %>% 
  dplyr::mutate(title=mean(c(start, end)))

# prepare a data frame for grid (scales)
grid_data <- base_data
grid_data$end <- grid_data$end[ c( nrow(grid_data), 1:nrow(grid_data)-1)] + 1
grid_data$start <- grid_data$start - 1
grid_data <- grid_data[-1,]

# Make the plot
prolific_sub1_p <- ggplot(data) +      
  
  # Add the stacked bar
  geom_bar(aes(x=as.factor(id), y=value, fill=observation), stat="identity", alpha=0.5)+
  scale_fill_viridis(discrete=TRUE, labels = c("m", "n", "f"), na.translate = F) +
  
  # Add a val=100/75/50/25 lines. I do it at the beginning to make sur barplots are OVER it.
  geom_segment(data=grid_data, aes(x = end, y = 0, xend = start, yend = 0), colour = "grey", alpha=1, linewidth=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 10, xend = start, yend = 10), colour = "grey", alpha=1, linewidth=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 20, xend = start, yend = 20), colour = "grey", alpha=1, linewidth=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 33, xend = start, yend = 33), colour = "grey", alpha=1, linewidth=0.3 , inherit.aes = FALSE ) +
  
  # Add text showing the value of each 100/75/50/25 lines
  ggplot2::annotate("text", x = rep(max(data$id),4), y = c(0, 10, 20, 33), label = c("0", "10", "20", "33") , color="grey", size=3 , angle=0, fontface="bold", hjust=1) + ylim(-50,max(label_data$tot, na.rm=T)) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(rep(-0,4), "cm"),
  ) +
  guides(fill = guide_legend(title = "Label")) + 
  coord_polar() +
  
  # Add labels on top of each bar
  geom_text(data=label_data, aes(x=id, y=tot-5, label=individual, hjust=hjust), color="black", fontface="bold",alpha=0.6, size=4, angle= label_data$angle, inherit.aes = FALSE) +
  # Add base line information
  geom_segment(data=base_data, aes(x = start, y = -5, xend = end, yend = -2), colour = "black", alpha=0.8, size=0.6 , inherit.aes = FALSE )  +
  geom_text(data=base_data, aes(x = title, y = -18, label=group), hjust=c(1,1,1,1), colour = "black", alpha=0.8, size=4, fontface="bold", inherit.aes = FALSE)

prolific_sub1_p
ggsave("results/fig_circular_barplot_prolificsub1.png", prolific_sub1_p, bg='transparent', width = 20, height = 20, units = "cm", dpi = 300)


########################################################
################ Association strength   ################
########################################################

### load data (not aggregated)
prolific_assoc_no_agg <- read.csv(file='data/human_annotators_long.csv', sep = ",")
prolific_assoc_no_agg <-  filter(prolific_assoc_no_agg, MFN != "n")
pos = which(names(prolific_assoc_no_agg) == "MFN")
names(prolific_assoc_no_agg)[pos] <- "Label"
pos = which(names(prolific_assoc_no_agg) == "Assoziation")
names(prolific_assoc_no_agg)[pos] <- "Association"
prolific_assoc_no_agg$Label <- ifelse(prolific_assoc_no_agg$Label == "w", "f", prolific_assoc_no_agg$Label)
prolific_assoc_no_agg$Label <- as.factor(prolific_assoc_no_agg$Label)
prolific_assoc_no_agg$Label <- factor(prolific_assoc_no_agg$Label, levels = c("m", "f"))


openAI_assoc_no_agg <- read.csv(file='data/openAI_results_long_df.csv', sep = ",")
openAI_assoc_no_agg <-  filter(openAI_assoc_no_agg, Label != "keins von beiden")
openAI_assoc_no_agg$Label <- ifelse(openAI_assoc_no_agg$Label == "Person männlichen Geschlechts", "m", "f")
openAI_assoc_no_agg$Label <- as.factor(openAI_assoc_no_agg$Label)
openAI_assoc_no_agg$Label <- factor(openAI_assoc_no_agg$Label, levels = c("m", "f"))
table(openAI_assoc_no_agg$Association, useNA = "always")



Llama_3.1_70B_assoc_no_agg <- read.csv(file='data/meta-llama-3.1-70b-instruct_ChatAI_results_long_df.csv', sep = ",")
Llama_3.1_70B_assoc_no_agg <-  filter(Llama_3.1_70B_assoc_no_agg, Label != "keins von beiden")
### sanity check und händische Korrektur falscher NAs
table(Llama_3.1_70B_assoc_no_agg$Association, useNA = "always")
which(is.na(Llama_3.1_70B_assoc_no_agg$Association))
Llama_3.1_70B_assoc_no_agg[which(is.na(Llama_3.1_70B_assoc_no_agg$Association)),]$Total_Output
Llama_3.1_70B_assoc_no_agg[which(is.na(Llama_3.1_70B_assoc_no_agg$Association)),]$Association <- 5
Llama_3.1_70B_assoc_no_agg$Label <- ifelse(Llama_3.1_70B_assoc_no_agg$Label == "Person männlichen Geschlechts", "m", "f")
Llama_3.1_70B_assoc_no_agg$Label <- as.factor(Llama_3.1_70B_assoc_no_agg$Label)
Llama_3.1_70B_assoc_no_agg$Label <- factor(Llama_3.1_70B_assoc_no_agg$Label, levels = c("m", "f"))


Llama_3.1_8B_assoc_no_agg <- read.csv(file='data/meta-llama-3.1-8b-instruct_ChatAI_results_long_df.csv', sep = ",")
Llama_3.1_8B_assoc_no_agg <-  filter(Llama_3.1_8B_assoc_no_agg, Label != "keins von beiden")
Llama_3.1_8B_assoc_no_agg$Label <- ifelse(Llama_3.1_8B_assoc_no_agg$Label == "Person männlichen Geschlechts", "m", "f")
### sanity check
table(Llama_3.1_8B_assoc_no_agg$Association, useNA = "always")
Llama_3.1_8B_assoc_no_agg_NAs = Llama_3.1_8B_assoc_no_agg[which(is.na(Llama_3.1_8B_assoc_no_agg$Association)),]
pos_digit = which(grepl("[[:digit:]]", Llama_3.1_8B_assoc_no_agg_NAs$Total_Output) == TRUE)
# View(Llama_3.1_8B_assoc_no_agg_NAs[pos_digit,])
Llama_3.1_8B_assoc_no_agg = Llama_3.1_8B_assoc_no_agg[-which(is.na(Llama_3.1_8B_assoc_no_agg$Association)),]
Llama_3.1_8B_assoc_no_agg$Label <- as.factor(Llama_3.1_8B_assoc_no_agg$Label)
Llama_3.1_8B_assoc_no_agg$Label <- factor(Llama_3.1_8B_assoc_no_agg$Label, levels = c("m", "f"))


### create histograms
library(ggplot2)
library(plyr)

color_m <- "#440154B3"
color_f <- "#FDE725B3"
colors_hist <- c(color_m, color_f)

### Human Annotators
mu <- ddply(prolific_assoc_no_agg, "Label", summarise, grp.mean=mean(Association))
hist_Human_Annotators <- ggplot(prolific_assoc_no_agg, aes(x=Association, fill = Label)) +
  geom_histogram(position="dodge", binwidth = 1)+
  scale_fill_manual(values = colors_hist) +
  scale_color_manual(values = colors_hist) +
  geom_vline(data=mu, aes(xintercept=grp.mean, color=Label),
             linetype="dashed") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 14),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="none"
  )  +
  labs(
    #subtitle = paste0("Human Annotators; N (total):", length(prolific_assoc_no_agg$Association), "; N (Label m): ", length(prolific_assoc_no_agg[which(prolific_assoc_no_agg$Label=="m"),]$Association), "; N (Label w): ",  length(prolific_assoc_no_agg[which(prolific_assoc_no_agg$Label=="w"),]$Association)),
    subtitle = paste0("Human Annotators; N: ", length(prolific_assoc_no_agg$Association)),
    x = "Association strength",
    y = "Count") +
  theme_light()
hist_Human_Annotators

### GPT4o
mu <- ddply(openAI_assoc_no_agg, "Label", summarise, grp.mean=mean(Association))
hist_openAI <- ggplot(openAI_assoc_no_agg, aes(x=Association, fill=Label)) +
  geom_histogram(position="dodge", binwidth = 1)+
  scale_fill_manual(values = colors_hist) +
  scale_color_manual(values = colors_hist) +
  geom_vline(data=mu, aes(xintercept=grp.mean, color=Label),
             linetype="dashed") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 14),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="none"
  )  +
  labs(
    # subtitle = paste0("GPT-4o; N (total):", length(openAI_assoc_no_agg$Association), "; N (Label m): ", length(openAI_assoc_no_agg[which(openAI_assoc_no_agg$Label=="m"),]$Association), "; N (Label w): ",  length(openAI_assoc_no_agg[which(openAI_assoc_no_agg$Label=="w"),]$Association)),
    subtitle = paste0("GPT-4o; N: ", length(openAI_assoc_no_agg$Association)),
    
    x = "Association strength",
    y = "Count") +
  theme_light()
hist_openAI

### Llama 70B
mu <- ddply(Llama_3.1_70B_assoc_no_agg, "Label", summarise, grp.mean=mean(Association))
hist_llama70B <- ggplot(Llama_3.1_70B_assoc_no_agg, aes(x=Association, fill=Label)) +
  geom_histogram(position="dodge", binwidth = 1)+
  scale_fill_manual(values = colors_hist) +
  scale_color_manual(values = colors_hist) +
  geom_vline(data=mu, aes(xintercept=grp.mean, color=Label),
             linetype="dashed") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 14),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="none"
  )  +
  labs(
    # subtitle = paste0("Llama 3.1 70B; N (total):", length(Llama_3.1_70B_assoc_no_agg$Association), "; N (Label m): ", length(Llama_3.1_70B_assoc_no_agg[which(Llama_3.1_70B_assoc_no_agg$Label=="m"),]$Association), "; N (Label w): ",  length(Llama_3.1_70B_assoc_no_agg[which(Llama_3.1_70B_assoc_no_agg$Label=="w"),]$Association)),
    subtitle = paste0("LLaMA 3.1 70B; N: ", length(Llama_3.1_70B_assoc_no_agg$Association)),
    x = "Association strength",
    y = "Count") +
  theme_light()
hist_llama70B


### Llama 8B
mu <- ddply(Llama_3.1_8B_assoc_no_agg, "Label", summarise, grp.mean=mean(Association))
hist_llama8B <- ggplot(Llama_3.1_8B_assoc_no_agg, aes(x=Association, fill=Label)) +
  geom_histogram(position="dodge", binwidth = 1)+
  scale_fill_manual(values = colors_hist) +
  scale_color_manual(values = colors_hist) +
  geom_vline(data=mu, aes(xintercept=grp.mean, color=Label),
             linetype="dashed") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 14),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="none"
  )  +
  labs(
    subtitle = paste0("LLaMA 3.1 8B; N: ", length(Llama_3.1_8B_assoc_no_agg$Association)),
    x = "Association strength",
    y = "Count") +
  theme_light()
hist_llama8B


### combine histograms
library(ggpubr)
library(grid)

hist_comb = ggarrange(hist_Human_Annotators + 
                          theme(axis.title.x = element_blank(),
                                axis.title.y = element_blank()),
                      hist_openAI + 
                          theme(axis.title.x = element_blank(),
                                axis.title.y = element_blank()),                    
                      hist_llama70B +
                          theme(axis.title.x = element_blank(),
                                axis.title.y = element_blank()),
                      hist_llama8B +
                          theme(axis.title.x = element_blank(),
                                axis.title.y = element_blank()),
                         nrow = 2,
                         ncol = 2,
                         common.legend = TRUE, legend="right")

title <- expression(atop(bold("Distribution of Association Strength")))
hist_comb = annotate_figure(hist_comb, top=textGrob(title, gp = gpar(fontsize = 14)), left = textGrob("Count", rot = 90, gp = gpar(fontsize = 14)), bottom = textGrob("Association Strength", gp = gpar(fontsize = 14)))
hist_comb

ggsave("results/fig_comb_histogram_assoc.png", hist_comb, bg='transparent', width = 20, height = 15, units = "cm", dpi = 300)



#################################################################
############### Association Strength and Logprobs  ##############
#################################################################

color_m <- "#440154B3"
color_f <- "#FDE725B3"
colors_hist <- c(color_m, color_f)

## GPT-4o
openAI_assoc_logprob_fig <- ggplot(openAI_assoc_no_agg, aes(x = Logprobs_Sum, y = Association, shape = Label, color = Label)) +
  geom_point(show.legend = FALSE) +
  geom_smooth(method = lm, aes(color = Label, fill = Label)) +  # Ensure both color & fill match
  scale_colour_manual(name = "Label", values = colors_hist) +  # Custom colors for points & lines
  scale_fill_manual(values = colors_hist, guide = "none") +  # Use same colors for fill but hide extra legend
  # labs(subtitle = paste0("GPT-4o; N (total):", length(openAI_assoc_no_agg$Association), "; N (Label m): ", length(openAI_assoc_no_agg[which(openAI_assoc_no_agg$Label=="m"),]$Association), "; N (Label w): ",  length(openAI_assoc_no_agg[which(openAI_assoc_no_agg$Label=="w"),]$Association)),
  labs(subtitle = paste0("GPT-4o; N (total): ", length(openAI_assoc_no_agg$Association)),
       x = "sum of subtoken logprobs per generated label",
       y = "association strength") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 12),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="bottom"
  ) +
  geom_rug() +
  theme_light()
openAI_assoc_logprob_fig


## Llama70B
llama70B_assoc_logprob_fig <- ggplot(Llama_3.1_70B_assoc_no_agg, aes(x = Logprobs_Sum, y = Association, shape = Label, color = Label)) +
  geom_point(show.legend = FALSE) +
  geom_smooth(method = lm, aes(color = Label, fill = Label)) +  # Ensure both color & fill match
  scale_colour_manual(name = "Label", values = colors_hist) +  # Custom colors for points & lines
  scale_fill_manual(values = colors_hist, guide = "none") +  # Use same colors for fill but hide extra legend
  labs(
    # subtitle = paste0("LLaMA 3.1 70B; N (total):", length(Llama_3.1_70B_assoc_no_agg$Association), "; N (Label m): ", length(Llama_3.1_70B_assoc_no_agg[which(Llama_3.1_70B_assoc_no_agg$Label=="m"),]$Association), "; N (Label w): ",  length(Llama_3.1_70B_assoc_no_agg[which(Llama_3.1_70B_assoc_no_agg$Label=="w"),]$Association)),
    subtitle = paste0("LLaMA 3.1 70B; N (total): ", length(Llama_3.1_70B_assoc_no_agg$Association)),
       x = "sum of subtoken logprobs per generated label",
       y = "association strength") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 12),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="bottom"
  ) +
  geom_rug() +
  theme_light()
llama70B_assoc_logprob_fig


## Llama8B
llama8B_assoc_logprob_fig <- ggplot(Llama_3.1_8B_assoc_no_agg, aes(x = Logprobs_Sum, y = Association, shape = Label, color = Label)) +
  geom_point(show.legend = FALSE) +
  geom_smooth(method = lm, aes(color = Label, fill = Label)) +  # Ensure both color & fill match
  scale_colour_manual(name = "Label", values = colors_hist) +  # Custom colors for points & lines
  scale_fill_manual(values = colors_hist, guide = "none") +  # Use same colors for fill but hide extra legend
  labs(
    # subtitle = paste0("Llama 3.1 8B; N (total):", length(Llama_3.1_8B_assoc_no_agg$Association), "; N (Label m): ", length(Llama_3.1_8B_assoc_no_agg[which(Llama_3.1_8B_assoc_no_agg$Label=="m"),]$Association), "; N (Label w): ",  length(Llama_3.1_8B_assoc_no_agg[which(Llama_3.1_8B_assoc_no_agg$Label=="w"),]$Association)),
    subtitle = paste0("LLaMA 3.1 8B; N (total): ", length(Llama_3.1_8B_assoc_no_agg$Association)),
       x = "sum of subtoken logprobs per generated label",
       y = "association strength") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 12),
    axis.title.x = element_text(size = 12),
    plot.title = element_text(size=14),
    legend.position="bottom"
  ) +
  geom_rug() +
  theme_light()
llama8B_assoc_logprob_fig


### combine assoc logprobs
library(ggpubr)
library(grid)

comb_assoc_logprob_fig = ggarrange(
                      openAI_assoc_logprob_fig + 
                        theme(axis.title.x = element_blank(),
                              axis.title.y = element_blank()),                    
                      llama70B_assoc_logprob_fig +
                        theme(axis.title.x = element_blank(),
                              axis.title.y = element_blank()),
                      llama8B_assoc_logprob_fig +
                        theme(axis.title.x = element_blank(),
                              axis.title.y = element_blank()),
                      nrow = 3,
                      ncol = 1,
                      common.legend = TRUE, legend="right")

title <- expression(atop(bold("Association Strength and Logprobs")))
comb_assoc_logprob_fig = annotate_figure(comb_assoc_logprob_fig, top=textGrob(title), left = "Association Strength", bottom = "Sum of Subtoken Logprobs per Generated Label")
comb_assoc_logprob_fig

ggsave("results/fig_comb_assoc_logprob.png", comb_assoc_logprob_fig, bg='transparent', width = 20, height = 15, units = "cm", dpi = 300)



########################################################
################# Agreement/Surprisal  #################
########################################################


color1 <- "#440154B3"
color2 <- "#21918cB3"
color3 <- "#FDE725B3"
colors = c(color1, color2, color3)
library(ggplot2)

OpenAI_agreement_logprobs_fig <- ggplot(gpt4o_res_comb_df, aes(x = logprobs_mean, y = agreement)) +
  geom_point(aes(color = maj_vote, fill = maj_vote, shape = maj_vote), size=2.5, show.legend = TRUE) +
  geom_smooth(method = lm, aes(color = maj_vote, fill = maj_vote), show.legend = FALSE) +  # Ensure both color & fill match
  scale_colour_manual(name = "Labels\nMajority Vote", values = colors) +  # Custom colors for points & lines
  scale_fill_manual(name = "Labels\nMajority Vote", values = colors) +  # Use same colors for fill but hide extra legend +
  scale_shape_manual(name = "Labels\nMajority Vote", values = c(15,16,17)) +
  labs(#title = "Label Agreement and Mean Surprisal",
    subtitle = paste0("GPT-4o; N = ", length(gpt4o_res_comb_df$np)),
    x = "mean surprisal per item label",
    y = "mean agreement per item") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 12),
    axis.title.x = element_text(size = 12),
    legend.position="none",
    plot.title = element_text(size=14)
  ) +
  geom_rug(aes(color = maj_vote), show.legend = FALSE) 
OpenAI_agreement_logprobs_fig


ChatAI_70B_agreement_logprobs_fig <- ggplot(ChatAI_70B_res_comb_df, aes(x = logprobs_mean, y = agreement)) +
  geom_point(aes(color = maj_vote, fill = maj_vote, shape = maj_vote), size=2.5, show.legend = TRUE) +
  geom_smooth(method = lm, aes(color = maj_vote, fill = maj_vote), show.legend = FALSE) +  # Ensure both color & fill match
  scale_colour_manual(name = "Labels\nMajority Vote", values = colors) +  # Custom colors for points & lines
  scale_fill_manual(name = "Labels\nMajority Vote", values = colors) +  # Use same colors for fill but hide extra legend +
  scale_shape_manual(name = "Labels\nMajority Vote", values = c(15,16,17)) +
  labs(#title = "Label Agreement and Mean Surprisal",
    subtitle = paste0("LLaMA 3.1 70B; N = ", length(ChatAI_70B_res_comb_df$np)),
    x = "mean surprisal per item label",
    y = "mean agreement per item") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 12),
    axis.title.x = element_text(size = 12),
    legend.position="none",
    plot.title = element_text(size=14)
  ) +
  geom_rug(aes(color = maj_vote), show.legend = FALSE) 
ChatAI_70B_agreement_logprobs_fig


ChatAI_8B_agreement_logprobs_fig <- ggplot(ChatAI_8B_res_comb_df, aes(x = logprobs_mean, y = agreement)) +
  geom_point(aes(color = maj_vote, fill = maj_vote, shape = maj_vote), size=2.5, show.legend = TRUE) +
  geom_smooth(method = lm, aes(color = maj_vote, fill = maj_vote), show.legend = FALSE) +  # Ensure both color & fill match
  scale_colour_manual(name = "Labels\nMajority Vote", values = colors) +  # Custom colors for points & lines
  scale_fill_manual(name = "Labels\nMajority Vote", values = colors) +  # Use same colors for fill but hide extra legend +
  scale_shape_manual(name = "Labels\nMajority Vote", values = c(15,16,17)) +
  labs(#title = "Label Agreement and Mean Surprisal",
    subtitle = paste0("LLaMA 3.1 8B; N = ", length(ChatAI_8B_res_comb_df$np)),
    x = "mean surprisal per item label",
    y = "mean agreement per item") +
  theme(
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.y = element_text(angle = 0, size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.text.x = element_text(angle = 0, size = 12),
    axis.title.x = element_text(size = 12),
    legend.position="bottom",
    plot.title = element_text(size=14)
  ) +
  geom_rug(aes(color = maj_vote), show.legend = FALSE) 
ChatAI_8B_agreement_logprobs_fig


library(ggpubr)
library(grid)

fig_comb_agreement_logprobs = ggarrange(OpenAI_agreement_logprobs_fig +
                                           theme(axis.title.x = element_blank(),
                                                 axis.title.y = element_blank()),
                                         ChatAI_70B_agreement_logprobs_fig + 
                                           theme(axis.title.x = element_blank(),
                                                 axis.title.y = element_blank()),
                                         ChatAI_8B_agreement_logprobs_fig + 
                                           theme(axis.title.x = element_blank(),
                                                 axis.title.y = element_blank()),
                                         nrow = 3,
                                         ncol = 1,
                                         common.legend = TRUE, legend="right")

title <- expression(atop(bold("Label Agreement and Mean Surprisal"), scriptstyle("N = 116 items")))
fig_comb_agreement_logprobs = annotate_figure(fig_comb_agreement_logprobs,
                                               top=textGrob(title), left = "mean agreement per item", bottom = "mean surpirsal by item")
fig_comb_agreement_logprobs
ggsave("results/fig_comb_agreement_logprobs_item.png", fig_comb_agreement_logprobs, bg='transparent', width = 20, height = 15, units = "cm", dpi = 300)




