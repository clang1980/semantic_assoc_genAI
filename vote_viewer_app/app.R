comb_scatter = read.csv("data/comb_scatter_df.csv")
euclidean_df = read.csv("data/comb_euclidean_df.csv")

np_filter = "Ahne"
subset(comb_scatter, np == np_filter)

library(shiny)
library(plotly)
library(bslib)
library(DT)
library(dplyr)

ui <- page_sidebar(
  title = "Label Distribution over 33 Trials",
  
  sidebar = sidebar(
    selectInput(
      inputId = "np_filter",
      label = "Select NP:",
      multiple = TRUE,
      choices = unique(comb_scatter$np),
      selected = "Ahne"
    ),
    br(),
    actionButton("info_btn", "More Info", icon = icon("info-circle"))
  ),
  
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  
  tags$head(
    tags$style(HTML("
      html, body {
        height: 100%;
        margin: 0;
      }
      #label_plot {
        height: calc(100vh - 100px) !important;
      }

      /* Smaller font in table */
      table.dataTable td, table.dataTable th {
        font-size: 12px !important;
        padding: 4px 6px !important;
      }

      /* Remove color from DataTable pagination buttons */
      .dataTables_wrapper .dataTables_paginate .paginate_button {
        background: none !important;
        border: none !important;
        color: inherit !important;
        box-shadow: none !important;
      }
      .dataTables_wrapper .dataTables_paginate .paginate_button:hover {
        background: none !important;
        text-decoration: underline;
      }
      .dataTables_wrapper .dataTables_paginate .paginate_button.current {
        font-weight: bold;
        text-decoration: underline;
      }
    "))
  ),
  
  layout_columns(
    col_widths = c(8, 4),  # 2/3 for plot, 1/3 for table
    
    card(
      full_screen = TRUE,
      card_header("3D Label Distribution Plot"),
      plotlyOutput("label_plot", height = "100%")
    ),
    
    card(
      card_header("Euclidean Distance from Human Annotators"),
      dataTableOutput("label_table")
    )
  )
)




server <- function(input, output) {
  output$label_plot <- renderPlotly({
    filtered_data <- subset(comb_scatter, np %in% input$np_filter)
    jittered_data <- filtered_data
    set.seed(123)  # For reproducibility
    jittered_data$m_jittered <- jitter(jittered_data$m, amount = 0)
    jittered_data$f_jittered <- jitter(jittered_data$f, amount = 0)
    jittered_data$n_jittered <- jitter(jittered_data$n, amount = 0)
    
    plot_ly(
      jittered_data,
      x = ~m_jittered,
      y = ~f_jittered,
      z = ~n_jittered,
      color = ~group,
      text = ~np,
      hovertext = paste(
        jittered_data$group,
        "<br> m :", jittered_data$m,
        "<br> f :", jittered_data$f,
        "<br> n :", jittered_data$n
      ),
      mode = "text"
    ) %>%
      add_markers(showlegend = FALSE, hoverinfo = "none") %>%
      add_text(hoverinfo = "text", textfont = list(size = 12)) %>%
      layout(
        scene = list(
          xaxis = list(title = 'person of male gender', range = c(-1, 34)),
          yaxis = list(title = 'person of female gender', range = c(-1, 34)),
          zaxis = list(title = 'neither', range = c(-1, 34))
        ),
        legend = list(
          orientation = "h",
          x = 0.5,
          y = -0.2,
          xanchor = "center",
          yanchor = "top"
        )
      )
  })

  
  observeEvent(input$info_btn, {
    showModal(modalDialog(
      title = "About This App",
      HTML("This interactive plot is supplementary material for the paper 'Using LLMs for experimental stimulus pretests in linguistics. Evidence from semantic associations between words and social gender', submitted to Konvens 2025. <br><br>
      The visualization allows you to explore the distribution of the 33 assigned labels per noun phrase (NP) evaluated for its association with social gender in a three-dimensional scatterplot. <br><br>
      The x-, y- and z-coordinates result from the number of labels assigned to the NP per label category ('person of male gender', 'person of female gender', 'neither'). 
      Hover over the labels to see the exact distribution. 
      To ensure readability we jittered the labels by 0.7 to prevent overlap.  <br> <br>
      The data table on the right displays the euclidean distance from the human annotators. <br> <br>
      
      Use the dropdown menu to select the NPs you wish to visualize—multiple selections are possible."),
      easyClose = TRUE,
      footer = NULL
    ))
  })
  
  output$label_table <- 
    renderDataTable({
      # Filter the data based on selected NP
      df_filtered <- subset(euclidean_df, np %in% input$np_filter)
      
      # Compute column means for the 3 numeric columns
      mean_row <- colMeans(df_filtered[, c("GPT.4o", "LlaMA_70B", "LlaMA_8B")])
      
      # Create a data frame with the mean row
      mean_row_df <- data.frame(
        np = "Mean",
        GPT.4o = mean_row["GPT.4o"],
        LlaMA_70B = mean_row["LlaMA_70B"],
        LlaMA_8B = mean_row["LlaMA_8B"]
      )
      
      # Combine filtered data with the mean row
      df_combined <- rbind(df_filtered, mean_row_df)
      
      # Render the DataTable
      datatable(
        df_combined,
        rownames = FALSE,  # Do not show row names
        options = list(
          paging = FALSE,
          info = FALSE,
          searching = FALSE
        )
      ) %>%
        formatRound(columns = c("GPT.4o", "LlaMA_70B", "LlaMA_8B"), digits = 3) %>%
        formatStyle(
          columns = names(df_combined)[1:4],  # Apply style to all visible columns
          target = 'row',
          fontWeight = styleEqual("Mean", "bold")
        )
    })  
  
}

shinyApp(ui, server)

