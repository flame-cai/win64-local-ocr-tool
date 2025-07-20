import random
import numpy as np
from typing import List, Tuple

from ..config.config_models import Config
from ..core.data_structures import GeneratedPage, TextBoxBlueprint, BoxType
from ..core.registries import LAYOUT_STRATEGIES
from ..utils.distribution_sampler import sample_from_config
from .content_generator import ContentGenerator
from ..augmentations import phase2_geometric, phase3_global
from ..core import coordinate_systems

# Add this import to load the registry correctly
from ..generators import layout_strategies

class PageGenerator:
    """
    Orchestrates the entire generation process for a single page.
    - Selects a layout strategy.
    - Generates content for textboxes.
    - Applies all augmentation phases.
    - Assembles the final page data.
    """
    def __init__(self, config: Config, random_state: random.Random):
        self.config = config
        self.random = random_state

    def generate_page(self) -> List[GeneratedPage]:
        """
        The main generation function.
        Returns a list of GeneratedPage objects. It's a list because
        ambiguous layouts produce multiple label interpretations for the
        same geometry. Standard layouts will return a list with one item.
        """
        seed = self.random.randint(0, 2**32 - 1)
        page_width = sample_from_config(self.config.page_dimensions['width'], self.random)
        page_height = sample_from_config(self.config.page_dimensions['height'], self.random)
        
        strategy_name = sample_from_config(self.config.layout_strategy_selection, self.random)
        strategy_func = LAYOUT_STRATEGIES[strategy_name]
        
        if strategy_name == "rejection_sampling":
            return [self._generate_standard_layout(strategy_func, page_width, page_height, seed)]
        else:
            return self._generate_ambiguous_layout(strategy_func, page_width, page_height, seed)

    def _generate_standard_layout(self, strategy_func, page_width, page_height, seed) -> GeneratedPage:
        blueprints = strategy_func(self.config, page_width, page_height, self.random)
        page_content = []
        aug_profile = self.config.augmentation_profiles['default']
        
        for blueprint in blueprints:
            box_config = self.config.textbox_types[blueprint.box_type.value]
            content_gen = ContentGenerator(box_config, aug_profile, self.random)
            textbox_obj = content_gen.generate_textbox_content(blueprint.width, blueprint.height, blueprint.box_type)

            points_local, line_ids_local = textbox_obj.consolidate()
            if points_local.shape[0] == 0: continue

            points_local_distorted = phase2_geometric.apply_phase2_augmentations(
                points_local, aug_profile, self.random
            )
            page_content.append((points_local_distorted, line_ids_local, blueprint))

        return self._assemble_page(page_content, page_width, page_height, blueprints, "rejection_sampling", seed, aug_profile_name="default")
    
    def _generate_ambiguous_layout(self, strategy_func, page_width, page_height, seed) -> List[GeneratedPage]:
        points_local, label_interpretations = strategy_func(self.config, page_width, page_height, self.random)
        aug_profile = self.config.augmentation_profiles['ambiguous']
        points_local_distorted = phase2_geometric.apply_phase2_augmentations(
            points_local, aug_profile, self.random
        )

        blueprint = TextBoxBlueprint(
            box_type=BoxType.AMBIGUOUS,
            position=(0, 0),
            width=np.max(points_local_distorted[:, 0]) - np.min(points_local_distorted[:, 0]) if points_local_distorted.shape[0] > 0 else 0,
            height=np.max(points_local_distorted[:, 1]) - np.min(points_local_distorted[:, 1]) if points_local_distorted.shape[0] > 0 else 0,
            orientation_deg=0
        )
        
        generated_pages = []
        for i, (label_name, line_ids) in enumerate(label_interpretations.items()):
            page_content = [(points_local_distorted, line_ids, blueprint)]
            page = self._assemble_page(page_content, page_width, page_height, [blueprint], f"ambiguous_{label_name}", seed, sub_id=i, aug_profile_name="ambiguous")
            generated_pages.append(page)
        
        return generated_pages

    def _assemble_page(self, page_content: list, page_width: float, page_height: float, blueprints: list, strategy_name: str, seed: int, sub_id: int = 0, aug_profile_name: str = "default") -> GeneratedPage:
        all_points_global, all_textbox_ids, all_textline_ids = [], [], []
        global_line_id_offset = 0

        for textbox_idx, (points_local, line_ids_local, blueprint) in enumerate(page_content):
            points_global = coordinate_systems.transform_to_global(points_local, blueprint)
            line_ids_global = line_ids_local + global_line_id_offset
            
            all_points_global.append(points_global)
            all_textline_ids.append(line_ids_global)
            all_textbox_ids.append(np.full(points_global.shape[0], textbox_idx))
            
            if len(line_ids_local) > 0:
                global_line_id_offset += (np.max(line_ids_local) + 1)

        if not all_points_global:
             return GeneratedPage(np.empty((0, 3)), np.empty((0,)), np.empty((0,)), blueprints, page_width, page_height, strategy_name, seed, sub_id)

        page_points = np.vstack(all_points_global)
        page_textbox_labels = np.concatenate(all_textbox_ids)
        page_textline_labels = np.concatenate(all_textline_ids)

        aug_profile = self.config.augmentation_profiles[aug_profile_name]

        # --- THIS IS THE CORRECTED CALL ---
        # We now pass all the required arguments to the function and unpack
        # the returned tuple into the final, synchronized variables.
        final_points, final_tb_labels, final_tl_labels = phase3_global.apply_phase3_augmentations(
            points=page_points,
            textbox_labels=page_textbox_labels,
            textline_labels=page_textline_labels,
            aug_profile=aug_profile,
            random_state=self.random
        )

        return GeneratedPage(
            points=final_points,
            textbox_labels=final_tb_labels,
            textline_labels=final_tl_labels,
            text_boxes=blueprints,
            page_width=page_width,
            page_height=page_height,
            layout_strategy=strategy_name,
            seed=seed,
            sub_id=sub_id,
        )