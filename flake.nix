# shells/cuda/flake.nix
{
  description = "CUDA development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      config.cudaSupport = true;
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      name = "cuda-env";

      buildInputs = with pkgs; [
        fish
        git
        uv
        ffmpeg
        
        # C/C++ Tools
        pkg-config
        cmake
        gnumake
        gcc
        
        # CUDA Tools
        cudatoolkit
        cudaPackages.cuda_cudart
        cudaPackages.cudnn
        
        # Graphics libs
        libGLU libGL
        xorg.libXi xorg.libXmu xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr
        freeglut
        zlib
        ncurses
      ];

      shellHook = ''
        # The Critical Fix: Add stdenv.cc.cc.lib to LD_LIBRARY_PATH
        export SHELL=${pkgs.fish}/bin/fish
        export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
        
        export CUDA_PATH=${pkgs.cudatoolkit}
        export EXTRA_LDFLAGS="-L/lib -L/run/opengl-driver/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
        
        echo "🚀 CUDA Environment Loaded"
      '';
    };
  };
}
