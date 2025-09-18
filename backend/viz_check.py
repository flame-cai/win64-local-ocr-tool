import os
import argparse
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def parse_page_xml(xml_file):
    """
    Parses a PAGE-XML file to extract TextRegion, TextLine coordinates, and Baselines.

    Args:
        xml_file (str): The path to the PAGE-XML file.

    Returns:
        dict: A dictionary containing the region polygons, line polygons, and baselines.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Namespace for PAGE-XML
    ns = {'page': 'https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    
    data = {
        'regions': [],
        'lines': [],
        'baselines': []
    }
    
    for text_region in root.findall('.//page:TextRegion', ns):
        # Extract TextRegion Coords
        region_coords = text_region.find('page:Coords', ns)
        if region_coords is not None:
            points_str = region_coords.get('points')
            points = [tuple(map(int, p.split(','))) for p in points_str.split()]
            data['regions'].append(points)
            
        for text_line in text_region.findall('page:TextLine', ns):
            # Extract TextLine Coords
            line_coords = text_line.find('page:Coords', ns)
            if line_coords is not None:
                points_str = line_coords.get('points')
                points = [tuple(map(int, p.split(','))) for p in points_str.split()]
                data['lines'].append(points)
            
            # Extract Baseline Coords
            baseline = text_line.find('page:Baseline', ns)
            if baseline is not None:
                points_str = baseline.get('points')
                points = [tuple(map(int, p.split(','))) for p in points_str.split()]
                data['baselines'].append(points)
                
    return data

def visualize_and_save(image_path, xml_data, output_path, visualize_region=False):
    """
    Overlays Baselines, Coords of every text line, and optionally the region 
    bounding polygon on an image and saves it.

    Args:
        image_path (str): The path to the input image.
        xml_data (dict): The parsed data from the corresponding PAGE-XML file.
        output_path (str): The path to save the visualized image.
        visualize_region (bool, optional): Whether to visualize the TextRegion 
                                           bounding polygon. Defaults to False.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Draw TextLine bounding polygons (Coords) in red
        for line_poly in xml_data['lines']:
            draw.polygon(line_poly, outline="red", width=2)

        # Draw Baselines in blue
        for baseline_points in xml_data['baselines']:
            draw.line(baseline_points, fill="blue", width=3)

        # Optionally draw TextRegion bounding polygon in green
        if visualize_region:
            for region_poly in xml_data['regions']:
                draw.polygon(region_poly, outline="green", width=3)

        image.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    """
    Main function to process images and XML files for visualization.
    """
    parser = argparse.ArgumentParser(description="Visualize PAGE-XML annotations on images.")
    parser.add_argument("image_dir", help="Directory containing the .jpg or .png images.")
    parser.add_argument("xml_dir", help="Directory containing the PAGE-XML files.")
    parser.add_argument("output_dir", help="Directory to save the visualized images.")
    parser.add_argument("--visualize_region", action="store_true", 
                        help="Visualize the TextRegion bounding polygon.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        base_name, _ = os.path.splitext(image_file)
        xml_file_path = os.path.join(args.xml_dir, base_name + ".xml")
        image_file_path = os.path.join(args.image_dir, image_file)
        output_file_path = os.path.join(args.output_dir, image_file)

        if os.path.exists(xml_file_path):
            xml_data = parse_page_xml(xml_file_path)
            visualize_and_save(image_file_path, xml_data, 
                               output_file_path, args.visualize_region)
            print(f"Processed and saved: {output_file_path}")
        else:
            print(f"Warning: No corresponding XML file found for {image_file}")

if __name__ == "__main__":
    main()

