# Face tracking reference intake

Place the user-supplied face photos in this directory. Keep the originals; the
commissioning step will create a cropped reference image and
`config/my-face.json` without modifying them.

The commissioned profile is loaded with:

```text
--face-detector-profile config/my-face.json
```

The profile should use the saved face crop for identity locking and enable
`live_bootstrap_frames` so the detector refreshes its appearance signature from
the first clean live frames under the current camera lighting. Until that file
exists, `switch to my face` selects the face target but `start tracking` remains
blocked with a clear “reference is not ready” status.
