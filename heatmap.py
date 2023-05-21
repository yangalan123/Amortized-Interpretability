import os

import thermostat
from datasets import load_dataset


def render(labels=False):
    """ Uses the displaCy visualization tool to render a HTML from the heatmap """

    # Call this function once for every text field
    if len(set([t.text_field for t in self])) > 1:
        for field in self[0].text_fields:
            print(f'Heatmap "{field}"')
            Heatmap([t for t in self if t.text_field == field]).render(labels=labels)
        return

    ents = []
    colors = {}
    ii = 0
    for color_token in self:
        ff = ii + len(color_token.token)

        # One entity in displaCy contains start and end markers (character index) and optionally a label
        # The label can be added by setting "attribution_labels" to True
        ent = {
            'start': ii,
            'end': ff,
            'label': str(color_token.score),
        }

        ents.append(ent)
        # A "colors" dict takes care of the mapping between attribution labels and hex colors
        colors[str(color_token.score)] = color_token.hex()
        ii = ff

    to_render = {
        'text': ''.join([t.token for t in self]),
        'ents': ents,
    }

    if labels:
        template = """
        <mark class="entity" style="background: {bg}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2;
        border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
            {text}
            <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform:
            uppercase; vertical-align: middle; margin-left: 0.5rem">{label}</span>
        </mark>
        """
    else:
        template = """
        <mark class="entity" style="background: {bg}; padding: 0.15em 0.3em; margin: 0 0.2em; line-height: 2.2;
        border-radius: 0.25em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
            {text}
        </mark>
        """

    html = displacy.render(
        to_render,
        style='ent',
        manual=True,
        jupyter=is_in_jupyter(),
        options={'template': template,
                 'colors': colors,
                 }
    )
    return html if not is_in_jupyter() else None


if __name__ == '__main__':
    seed_1_path = "path/to/thermostat/experiments/thermostat/multi_nli/bert/svs-2000/seed_1/[date1].ShapleyValueSampling.jsonl"
    seed_2_path = "path/to/thermostat/experiments/thermostat/multi_nli/bert/svs-2000/seed_2/[date2].ShapleyValueSampling.jsonl"
    ds_1 = load_dataset("json", data_files=[seed_1_path, ])['train']
    ds_1._info.description = "Model: textattack/bert-base-uncased-MNLI\nDataset: MNLI\nExplainer: svs-2000"
    ds_1 = ds_1.add_column("idx", list(range(len(ds_1))))
    obj_1 = thermostat.Thermopack(ds_1)
    target_dir = "visualization/heatmap"
    os.makedirs(target_dir, exist_ok=True)
    for i in range(100):
        img1 = data_1[i].render()
        f_1 = open(os.path.join(target_dir, f"{i}_seed_1.html"), "w", encoding='utf-8')
        f_1.write(img1)
        img2 = data_2[i].render()
        f_2 = open(os.path.join(target_dir, f"{i}_seed_2.html"), "w", encoding='utf-8')
        f_2.write(img2)
