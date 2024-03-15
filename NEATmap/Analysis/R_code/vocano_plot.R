library(ggplot2)
setwd("R:/WeijieZheng/Social_defeat_csv")

data <- read.csv("I_O_difference_level6.csv")
ggplot(data, aes(log2FoldChange, -log10(pValue)))+
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "#999999")+
  
  geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", color = "#999999")+
  
  geom_point(aes(size=6, color=-log10(pValue)))+
  
  geom_text(aes(label=label, color = -log10(pValue)), size=4, vjust = -0.7, hjust=1)+
  
  scale_color_gradientn(values = seq(0, 1, 0.2),
                        colors = c("#39489f", "#39bbec", "#f9ed36", "#f38466", "#b81f25"))+
  
  scale_size_continuous((range = c(1,3)))+
  labs(title = "SDS vs ES")+
  theme_bw()+
  theme(panel.grid = element_blank(), 
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        plot.title = element_text(hjust = 0.5, size = 18),
        legend.position = c(0.764, 0.5),
        legend.justification = c(0,1),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 18)
        )+
  
  guides(col = guide_colorbar(title = "-Lg(q-value)"),
         size = "none")+
  
  xlab(expression(Log[2](FoldChange)))+
  ylab("-Lg(FDR q-value)")

ggsave("vocanol_plot_sds_vs_es.pdf", height = 6, width = 9)
