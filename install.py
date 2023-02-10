import launch

deforum_libs = ["pandas", "matplotlib", "numexpr", "av", "pims", "moviepy", "imageio_ffmpeg"]
for lib in deforum_libs:
    if not launch.is_installed(lib):
        launch.run_pip(f"install {lib}", f"deforum requirement: {lib}")