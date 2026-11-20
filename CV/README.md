# Camera drawing

Install the dependencies from `CV/requirements.txt` and `ML/requirements.txt`.
From the repository root, run each command in its own terminal:

```sh
python CV/server.py
streamlit run ML/UI/canvas.py
```

Open [Streamlit](http://localhost:8501), select **Camera**, and click
**Connect camera**. This opens the camera preview on the machine running the
camera server. Keep the canvas clear during calibration. Draw a blue line,
then move your hand out of the camera view. The camera traces the new ink
after eight frames without a detected arm and sends the line to Streamlit
for a model continuation. Use **Clear** to reset the Streamlit drawing or
**Disconnect camera** to close the connection.

Streamlit previews the generated G-code and sends each new continuation to the
plotter once. Enable **Practice run (pretend Arduino)** before drawing to test
the workflow without moving the plotter. Select **Draw in Streamlit** to use the
mouse canvas.
