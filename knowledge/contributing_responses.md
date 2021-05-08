# Contributing Responses

## Response Style Guide

TODO: Currently there is no strict style guide for responses.

However, a good example is [Can pets be vegan?](https://github.com/veganhacktivists/animalsupportbot/blob/master/knowledge/responses/can_pets_be_vegan.md)

## Modifying an existing response

The responses are located in [`knowledge/responses`](https://github.com/veganhacktivists/animalsupportbot/tree/master/knowledge/responses) in `.md` files.

The easiest way to modify an existing response is via GitHub. You must be logged into GitHub to do this. 

On the response `.md` file which you wish to edit:
   1. Click "Edit this file" button around the top right of the file text - this looks like a pen/pencil shaped icon for most.
   2. Make your changes in the editing box.
   3. Describe the changes you made in a commit message at the bottom.
   4. Click the "Propose changes" button at the bottom.

After this, a maintainer of this project will approve/reject the changes you made.

TODO: video tutorial how to propose changes to a response.

## Adding a new myth/response

In order to add a new myth/response, two files must be created.

  1. `knowledge/myths/<myth>.yaml`
  2. `knowledge/responses/<myth>.md`

### YAML File

This file contains the general information needed to match and catalogue this myth. This is covered in the repo `README.md` but is covered again here, with "Plants feel Pain" as an example:

```yaml
key: plants_feel_pain
title: Plants Feel Pain 
full_comment: false 
link: <URL> 
examples:
- what if plants feel pain
- plants feel pain too
- how do you know plants don't feel pain
```

- `key` is the unique identifier for this argument. The response text in `knowledge/responses` must have the filename: `<key>.md`.
- `title` is the formatted title for this argument.
- `full_comment` is a boolean which indicates whether or not the full response should be posted. If this is `false` then the most similar sentence to the input in the response text is selected (along with the proceeding 5 sentences). If this is `false` then there should be no markdown formatting as this is currently not supported.
- `link` an optional link to highlight the argument title with in the response, such as a YouTube video. If there is no link, this must be set to `nan`.
- `examples` the example sentences/phrases which should link to this argument. These examples make up the "training set" for the nearest neighbor classifier. Try and add as many varied examples as you can think of for your myth/argument.

### Response .md File

This file is where the response text is taken from. It must have the filename `<key>.md` specified in the YAML file above.

If `full_comment` is set to `true` in the YAML file, this can use the [Reddit markdown](https://reddit.com/wiki/markdown) formatting.
