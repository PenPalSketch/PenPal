# Camera drawing

Install the dependencies from `CV/requirements.txt` and `ML/requirements.txt`.
From the repository root, run each command in its own terminal:

```sh
python CV/server.py
streamlit run ML/UI/canvas.py
```

Open [Streamlit](http://localhost:8501), select **Camera**, and click
**Connect camera**. This opens the camera preview on the machine running the
camera server. Each pen lift triggers a model continuation. Use **Clear** to
reset the drawing or **Disconnect camera** to close the connection.

Streamlit displays the plotter preview, G-code download, and **Send to plotter**
button. Select **Draw in Streamlit** to use the mouse canvas.
