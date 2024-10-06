# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""DocRED"""

from __future__ import absolute_import, division, print_function

import json
import logging

import datasets

import math
from collections import defaultdict

_DESCRIPTION = """\
DocRED for RE.
"""

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}

mapping_types = {
    "Researcher": "<Researcher>",
    "Material": "<Material>",
    "Writer": "<Writer>",
    "Chemical element": "<Chemical element>",
    "Magazine": "<Magazine>",
    "Policital party": "<Policital party>",
    "Album": "<Album>",
    "Location": "<Location>",
    "Musical instrument": "<Musical instrument>",
    "Discipline": "<Discipline>",
    "University": "<University>",
    "Music genre": "<Music genre>",
    "Time": "<Time>",
    "Politician": "<Politician>",
    "Organization": "<Organization>",
    "Literary genre": "<Literary genre>",
    "Band": "<Band>",
    "Song": "<Song>",
    "Field": "<Field>",
    "Election": "<Election>",
    "Enzyme": "<Enzyme>",
    "Programming language": "<Programming language>",
    "Metric": "<Metric>",
    "Other": "<Other>",
    "Person": "<Person>",
    "Artist": "<Artist>",
    "Academic journal": "<Academic journal>",
    "Algorithm": "<Algorithm>",
    "Scientist": "<Scientist>",
    "Event": "<Event>",
    "Book": "<Book>",
    "Poem": "<Poem>",
    "Astronomical object": "<Astronomical object>",
    "Method": "<Method>",
    "Country": "<Country>",
    "Theory": "<Theory>",
    "Number": "<Number>",
    "Task": "<Task>",
    "Award": "<Award>",
    "Product": "<Product>",
    "Scientific term": "<Scientific term>",
    "Conference": "<Conference>",
    "Protein": "<Protein>",
}


class DocREDConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DocREDConfig, self).__init__(**kwargs)


class DocRED(datasets.GeneratorBasedBuilder):
    """DocRED Version 1.0."""

    BUILDER_CONFIGS = [
        DocREDConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/lavis-nlp/jerex/",
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"],
                "dev": self.config.data_files["dev"],
                "test": self.config.data_files["test"],
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        print("generating examples from = %s", filepath)
        with open(filepath) as json_file:
            f = json.load(json_file)
            for id_, row in enumerate(f):
                triplets = ""
                prev_head = None
                relations_sorted = sorted(row["labels"], key=lambda tup: tup["h"])
                for relation in relations_sorted:
                    if prev_head == relation["h"]:
                        triplets += (
                            f' {mapping_types[row["vertexSet"][relation["h"]][0]["type"]]} '
                            + row["vertexSet"][relation["t"]][0]["name"]
                            + f' {mapping_types[row["vertexSet"][relation["t"]][0]["type"]]} '
                            + relation["r"]
                        )
                    elif prev_head == None:
                        triplets += (
                            "<triplet> "
                            + row["vertexSet"][relation["h"]][0]["name"]
                            + f' {mapping_types[row["vertexSet"][relation["h"]][0]["type"]]} '
                            + row["vertexSet"][relation["t"]][0]["name"]
                            + f' {mapping_types[row["vertexSet"][relation["t"]][0]["type"]]} '
                            + relation["r"]
                        )
                        prev_head = relation["h"]
                    else:
                        triplets += (
                            " <triplet> "
                            + row["vertexSet"][relation["h"]][0]["name"]
                            + f' {mapping_types[row["vertexSet"][relation["h"]][0]["type"]]} '
                            + row["vertexSet"][relation["t"]][0]["name"]
                            + f' {mapping_types[row["vertexSet"][relation["t"]][0]["type"]]} '
                            + relation["r"]
                        )
                        prev_head = relation["h"]
                # print(str(row["title"]), {
                #     "title": str(row["title"]),
                #     "context": ' '.join([word for sentence in row['sents'] for word in sentence]),
                #     "id": str(row["title"]),
                #     "triplets": triplets,
                # })
                yield (row["title"]), {
                    "title": str(row["title"]),
                    "context": " ".join(
                        [word for sentence in row["sents"] for word in sentence]
                    ),
                    "id": str(row["title"]),
                    "triplets": triplets,
                }
