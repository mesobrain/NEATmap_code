# Usage: Rscript vocano_plot.R [input.csv] [output.pdf]
# source() defines plot_volcano() without changing directories or plotting.
plot_volcano <- function(input_file = "I_O_difference_level6.csv",
                         output_file = "vocanol_plot_sds_vs_es.pdf",
                         title = "SDS vs ES", width = 9, height = 6) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required to draw the volcano plot.", call. = FALSE)
  }
  data <- utils::read.csv(input_file, stringsAsFactors = FALSE, check.names = FALSE,
                          na.strings = c("", "NA", "NaN"))
  if (anyDuplicated(names(data))) {
    stop("Input CSV contains duplicate column names.", call. = FALSE)
  }
  probability_column <- if ("qValue" %in% names(data)) "qValue" else "pValue"
  required <- c("log2FoldChange", probability_column)
  missing_columns <- setdiff(required, names(data))
  if (length(missing_columns)) {
    stop(paste("Missing required columns:", paste(missing_columns, collapse = ", ")),
         call. = FALSE)
  }
  for (column in required) {
    original <- data[[column]]
    numeric_values <- suppressWarnings(as.numeric(as.character(original)))
    if (any(!is.na(original) & is.na(numeric_values))) {
      stop(paste("Column must contain numeric values:", column), call. = FALSE)
    }
    data[[column]] <- numeric_values
  }
  probabilities <- data[[probability_column]]
  if (any(!is.na(probabilities) &
          (!is.finite(probabilities) | probabilities < 0 | probabilities > 1))) {
    stop(paste(probability_column, "must be between 0 and 1."), call. = FALSE)
  }
  keep <- is.finite(data$log2FoldChange) & !is.na(probabilities)
  if (any(!keep)) {
    warning(sprintf("Omitting %d rows with missing or non-finite plot values.",
                    sum(!keep)), call. = FALSE)
  }
  data <- data[keep, , drop = FALSE]
  if (!nrow(data)) {
    stop("No valid rows remain for the volcano plot.", call. = FALSE)
  }
  if (!"label" %in% names(data)) {
    data$label <- ""
  }
  data$label <- as.character(data$label)
  data$label[is.na(data$label)] <- ""
  # Zero may result from underflow; cap only the plotted transform.
  data$.neg_log10 <- -log10(pmax(data[[probability_column]], .Machine$double.xmin))
  value_label <- if (probability_column == "qValue") "FDR q-value" else "p-value"
  plot <- ggplot2::ggplot(data, ggplot2::aes(log2FoldChange, .neg_log10)) +
    ggplot2::geom_hline(yintercept = -log10(0.05), linetype = "dashed",
                        colour = "#999999") +
    ggplot2::geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed",
                        colour = "#999999") +
    ggplot2::geom_point(ggplot2::aes(colour = .neg_log10), size = 3) +
    ggplot2::geom_text(data = data[nzchar(data$label), , drop = FALSE],
                       ggplot2::aes(label = label, colour = .neg_log10),
                       size = 4, vjust = -0.7, hjust = 1) +
    ggplot2::scale_colour_gradientn(
      colours = c("#39489f", "#39bbec", "#f9ed36", "#f38466", "#b81f25")) +
    ggplot2::labs(title = title, x = expression(Log[2](FoldChange)),
                  y = paste0("-log10(", value_label, ")"),
                  colour = paste0("-log10(", value_label, ")")) +
    ggplot2::theme_bw() +
    ggplot2::theme(panel.grid = ggplot2::element_blank(),
                   axis.text = ggplot2::element_text(size = 16),
                   axis.title = ggplot2::element_text(size = 18),
                   plot.title = ggplot2::element_text(hjust = 0.5, size = 18),
                   legend.justification = c(0, 1),
                   legend.text = ggplot2::element_text(size = 16),
                   legend.title = ggplot2::element_text(size = 18))
  if (utils::packageVersion("ggplot2") >= "3.5.0") {
    plot <- plot + ggplot2::theme(legend.position = "inside",
                                  legend.position.inside = c(0.764, 0.5))
  } else {
    plot <- plot + ggplot2::theme(legend.position = c(0.764, 0.5))
  }
  if (!is.null(output_file)) {
    ggplot2::ggsave(output_file, plot = plot, height = height, width = width)
  }
  invisible(plot)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) > 2L) {
    stop("Usage: Rscript vocano_plot.R [input.csv] [output.pdf]", call. = FALSE)
  }
  do.call(plot_volcano, as.list(args))
}
