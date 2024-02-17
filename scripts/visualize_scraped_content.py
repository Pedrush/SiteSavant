import json
import logging


def main():
    """Loads scraped data from a JSON file and converts it into a markdown file format for easier
    visualization.

    Inputs:
        A JSON file containing the scraped data.

    Outputs:
        A markdown file at the specified output path. This file contains the formatted data from
        the input JSON, designed for easy reading and debugging of scraped data.
    """
    input_json_file_path = "data/scraped_data/scraped_data_08-02-2024_21_50_37.json"
    output_markdown_file_path = "data/debug/scraped_text_visualization.md"

    try:
        with open(input_json_file_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {input_json_file_path}")
        data = []

    with open(output_markdown_file_path, "w") as file:
        for item in data:
            file.write(f"## [{item['title']}]({item['url']})\n\n")
            file.write(f"**URL:** {item['url']}\n\n")
            file.write(f"**Scrape Timestamp:** {item.get('scrape_timestamp', 'N/A')}\n\n")
            file.write(f"**Text:**\n\n{item['text']}\n\n")
            file.write("---\n\n")  # Separator between entries

    logging.info(f"Markdown file created at: {output_markdown_file_path}")


if __name__ == "__main__":
    main()
