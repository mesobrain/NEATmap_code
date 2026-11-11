# Usage: Rscript heatmap_bupple.R [input.csv] [output.pdf]
# Numeric pvalue data use a continuous scale without inventing significance bins.
# Use pvalue_type = "categorical" for explicitly pre-binned numeric categories.
plot_correlation_bubbles <- function(input_file = "sds_es_corr.csv",
                                      output_file = "person_corr.pdf",
                                      pvalue_type = c("auto", "categorical", "continuous"),
                                      pvalue_label = "p-value", width = 18, height = 2) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required to draw the correlation plot.", call. = FALSE)
  }
  pvalue_type <- match.arg(pvalue_type)
  data <- utils::read.csv(input_file, stringsAsFactors = FALSE, check.names = FALSE,
                          na.strings = c("", "NA", "NaN"))
  if (anyDuplicated(names(data))) {
    stop("Input CSV contains duplicate column names.", call. = FALSE)
  }
  missing_columns <- setdiff(c("Region", "group", "corr", "pvalue"), names(data))
  if (length(missing_columns)) {
    stop(paste("Missing required columns:", paste(missing_columns, collapse = ", ")),
         call. = FALSE)
  }
  original_corr <- data$corr
  data$corr <- suppressWarnings(as.numeric(as.character(original_corr)))
  if (any(!is.na(original_corr) & is.na(data$corr))) {
    stop("Column 'corr' must contain numeric values.", call. = FALSE)
  }
  if (any(!is.na(data$corr) & (!is.finite(data$corr) | abs(data$corr) > 1))) {
    stop("Column 'corr' must be between -1 and 1.", call. = FALSE)
  }
  continuous <- pvalue_type == "continuous" ||
    (pvalue_type == "auto" && is.numeric(data$pvalue))
  if (continuous) {
    original_pvalue <- data$pvalue
    data$pvalue <- suppressWarnings(as.numeric(as.character(original_pvalue)))
    if (any(!is.na(original_pvalue) & is.na(data$pvalue))) {
      stop("Continuous pvalue data must be numeric.", call. = FALSE)
    }
    if (any(!is.na(data$pvalue) &
            (!is.finite(data$pvalue) | data$pvalue < 0 | data$pvalue > 1))) {
      stop("Numeric pvalue data must be between 0 and 1.", call. = FALSE)
    }
  } else {
    data$pvalue <- as.character(data$pvalue)
  }
  keep <- !is.na(data$Region) & nzchar(trimws(as.character(data$Region))) &
    !is.na(data$group) & nzchar(trimws(as.character(data$group))) &
    !is.na(data$corr) & !is.na(data$pvalue)
  if (!continuous) {
    keep <- keep & nzchar(trimws(data$pvalue))
  }
  if (any(!keep)) {
    warning(sprintf("Omitting %d rows with missing plot values.", sum(!keep)),
            call. = FALSE)
  }
  data <- data[keep, , drop = FALSE]
  if (!nrow(data)) {
    stop("No valid rows remain for the correlation plot.", call. = FALSE)
  }
  data$Region <- factor(data$Region, levels = unique(data$Region))
  data$group <- factor(data$group, levels = unique(data$group))
  negative_colours <- c("#212c5f", "#3366b1", "#42b0e4", "#dfe1e0")
  positive_colours <- c("#f26666", "#f49699", "#facccc", "#d9dbd9")
  if (!continuous) {
    # Named palettes prevent a category changing colour when a sign lacks it.
    categories <- sort(unique(data$pvalue))
    if (length(categories) > length(negative_colours)) {
      stop("Categorical pvalue data may contain at most four categories.", call. = FALSE)
    }
    data$pvalue <- factor(data$pvalue, levels = categories)
    negative_colours <- stats::setNames(negative_colours[seq_along(categories)], categories)
    positive_colours <- stats::setNames(positive_colours[seq_along(categories)], categories)
  }
  plot <- ggplot2::ggplot(data, ggplot2::aes(Region, group, size = abs(corr))) +
    ggplot2::geom_point(data = data[data$corr <= 0, , drop = FALSE],
                        ggplot2::aes(fill = pvalue), colour = "#999999", shape = 21) +
    ggplot2::geom_point(data = data[data$corr > 0, , drop = FALSE],
                        ggplot2::aes(colour = pvalue), shape = 16) +
    ggplot2::scale_size_area(max_size = 6, limits = c(0, 1)) +
    ggplot2::scale_x_discrete(drop = FALSE) +
    ggplot2::scale_y_discrete(drop = FALSE) +
    ggplot2::theme_bw() +
    ggplot2::theme(legend.position = "bottom",
                   panel.grid.major.x = ggplot2::element_blank(),
                   panel.grid.minor.x = ggplot2::element_blank(),
                   legend.margin = ggplot2::margin(5, unit = "pt"),
                   axis.text.y = ggplot2::element_text(angle = 90, hjust = 0.5),
                   axis.text = ggplot2::element_text(size = 12)) +
    ggplot2::labs(x = "", y = "", size = "Absolute Spearman correlation",
                  fill = paste("Negative correlation", pvalue_label),
                  colour = paste("Positive correlation", pvalue_label)) +
    ggplot2::guides(size = ggplot2::guide_legend(order = 1,
                                                override.aes = list(shape = 21)))
  if (continuous) {
    plot <- plot +
      ggplot2::scale_fill_gradientn(colours = negative_colours, limits = c(0, 1)) +
      ggplot2::scale_colour_gradientn(colours = positive_colours, limits = c(0, 1)) +
      ggplot2::guides(fill = ggplot2::guide_colourbar(order = 2),
                      colour = ggplot2::guide_colourbar(order = 3))
  } else {
    plot <- plot +
      ggplot2::scale_fill_manual(values = negative_colours, limits = categories, drop = FALSE) +
      ggplot2::scale_colour_manual(values = positive_colours, limits = categories, drop = FALSE) +
      ggplot2::guides(fill = ggplot2::guide_legend(order = 2,
                                                 override.aes = list(size = 4)),
                      colour = ggplot2::guide_legend(order = 3,
                                                   override.aes = list(size = 4)))
  }
  if (!is.null(output_file)) {
    ggplot2::ggsave(output_file, plot = plot, height = height, width = width)
  }
  invisible(plot)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) > 2L) {
    stop("Usage: Rscript heatmap_bupple.R [input.csv] [output.pdf]", call. = FALSE)
  }
  do.call(plot_correlation_bubbles, as.list(args))
}
