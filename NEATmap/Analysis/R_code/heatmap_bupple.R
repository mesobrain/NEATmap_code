library(ggplot2)
library(readr)
setwd('R:/WeijieZheng/Social_defeat_csv')

data <- read_csv('sds_es_corr.csv', show_col_types = FALSE)
ggplot()+
  geom_point(data = data, aes(x = Region, y = group, size=abs(corr), fill=pvalue), color="#999999", shape=21)+
  geom_point(data = data[which(data$neg_pos == "Pos"), ],aes(x = Region, y = group, size=abs(corr), color=pvalue), shape=16)+
  scale_fill_manual(values = c("#212c5f", "#3366b1", "#42b0e4", "#dfe1e0"))+
  scale_color_manual(values = c("#f26666", "#f49699", "#facccc", "#d9dbd9"))+
  theme_bw()+
  theme(legend.position = "bottom",
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        legend.margin = margin(5, unit = "pt"),
        axis.text.y = element_text(angle = 90, hjust = 0.5),
        axis.text =  element_text(size = 12))+
  xlab("")+
  ylab("")+
  guides(size = guide_legend(title = expression("Spearman correlation"), order = 1,
                             override.aes = list(shape=21)),
         fill = guide_legend(title = expression("Negative correlation FDR q-value"), order = 2,
                             override.aes = list(size = 4)),
         col = guide_legend(title = expression("Postive correlation FDR q-value"), order = 3,
                            override.aes = list(size = 4)))
ggsave("person_corr.pdf", height = 2, width = 18)
