from utilities import *

global conn, headers
product_mapping = {}

## Gradio Interface
with gr.Blocks() as demo:

    gr.Markdown("## Amazon Product Search")
    gr.Markdown("Enter a search query, select a product, and confirm your choice.")

    # Input field for search query
    search_query = gr.Textbox(label="Search Query", placeholder="e.g., iPhone 12")

    # Dropdown for displaying search results dynamically
    search_results = gr.Dropdown(label="Search Results", choices=[], interactive=True)

    # Button to trigger search
    search_button = gr.Button("Search")

    # Output for showing selected product confirmation
    selected_output = gr.Textbox(label="Average rating and standard deviation based on product reviews", interactive=False)

    chart_output = gr.HTML(label="Rating Visualization")

    with gr.Row():  # Organize buttons and logs
        confirm_button = gr.Button("Confirm Selection")

    # Add spacing before the log output
    gr.Markdown("<br>")  # Add space between the button and logs

    # Log output box placed below the confirm button
    log_output = gr.Textbox(label="Debug Logs", interactive=False, lines=3)

    # Event to search products and update dropdown
    confirm_button.click(
        fn=process_selection,
        inputs=search_results,
        outputs=[selected_output, chart_output]
    )

    search_button.click(
        fn=process_search,
        inputs=search_query,
        outputs=[search_results, log_output],
    )

# Launch the interface
demo.launch()
