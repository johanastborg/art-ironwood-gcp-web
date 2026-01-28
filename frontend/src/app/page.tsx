"use client";

import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [rendering, setRendering] = useState(false);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [sceneObjects, setSceneObjects] = useState<any[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        // Clear background
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw grid
        ctx.strokeStyle = "#333";
        for(let i=0; i<canvas.width; i+=50) {
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, canvas.height);
            ctx.stroke();
        }
        for(let i=0; i<canvas.height; i+=50) {
            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(canvas.width, i);
            ctx.stroke();
        }

        // Draw objects
        ctx.fillStyle = "#4f9";
        sceneObjects.forEach(obj => {
           ctx.beginPath();
           ctx.arc(obj.x, obj.y, 20, 0, 2 * Math.PI);
           ctx.fill();
        });
      }
    }
  }, [sceneObjects]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          setSceneObjects([...sceneObjects, { type: "sphere", x, y, radius: 20 }]);
      }
  };

  const handleRender = async () => {
    setRendering(true);
    setResultImage(null);
    try {
      // In a real app, URL should be env var
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/render`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          scene: { objects: sceneObjects },
          width: 800,
          height: 600,
          samples: 128
        }),
      });

      if (!response.ok) {
        throw new Error("Render failed");
      }

      const data = await response.json();

      if (data.status === "completed" && data.tiles) {
        // Reconstruct image from tiles
        const resultCanvas = document.createElement("canvas");
        resultCanvas.width = 800;
        resultCanvas.height = 600;
        const ctx = resultCanvas.getContext("2d");

        if (ctx) {
            for (let index = 0; index < data.tiles.length; index++) {
                const tile = data.tiles[index];

                // 600 / 8 = 75
                const stripHeight = 600 / 8;
                const y = index * stripHeight;

                // Decode base64 to Uint8Array (raw RGB bytes)
                const raw = atob(tile);
                const uint8Array = new Uint8Array(raw.length);
                for (let i = 0; i < raw.length; i++) {
                    uint8Array[i] = raw.charCodeAt(i);
                }

                const width = 800;
                const height = stripHeight;
                const rgbaData = new Uint8ClampedArray(width * height * 4);

                for (let i = 0; i < width * height; i++) {
                    rgbaData[i * 4 + 0] = uint8Array[i * 3 + 0]; // R
                    rgbaData[i * 4 + 1] = uint8Array[i * 3 + 1]; // G
                    rgbaData[i * 4 + 2] = uint8Array[i * 3 + 2]; // B
                    rgbaData[i * 4 + 3] = 255;                   // A
                }

                const imageData = new ImageData(rgbaData, width, height);
                ctx.putImageData(imageData, 0, y);
            }
            setResultImage(resultCanvas.toDataURL());
        }
      }
    } catch (error) {
      console.error(error);
      alert("Render failed. Make sure backend is running.");
    } finally {
      setRendering(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-8 bg-zinc-900 text-white font-sans">
      <h1 className="text-4xl font-bold mb-8">Art Ironwood: Distributed Renderer</h1>

      <div className="flex gap-8">
        <div className="flex flex-col gap-4">
          <h2 className="text-xl font-semibold">Scene Layout</h2>
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="border border-zinc-700 cursor-crosshair bg-black"
            onClick={handleCanvasClick}
          />
          <p className="text-zinc-400 text-sm">Click to add objects</p>
        </div>

        <div className="flex flex-col gap-4">
          <h2 className="text-xl font-semibold">Render Result</h2>
          <div className="w-[800px] h-[600px] border border-zinc-700 bg-black flex items-center justify-center">
            {resultImage ? (
               <img src={resultImage} alt="Rendered Scene" width={800} height={600} />
            ) : rendering ? (
               <div className="text-zinc-400 animate-pulse">Rendering on GCP Farm...</div>
            ) : (
               <div className="text-zinc-600">No render yet</div>
            )}
          </div>
        </div>
      </div>

      <button
        onClick={handleRender}
        disabled={rendering}
        className="mt-8 px-8 py-4 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 rounded-lg font-bold text-lg transition-colors"
      >
        {rendering ? "Processing..." : "Render Scene"}
      </button>
    </div>
  );
}
