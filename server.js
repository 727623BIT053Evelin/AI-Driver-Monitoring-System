const http = require("http");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

let pythonProcess = null;
let pythonWindowId = null; // Track the Python window for focus

const server = http.createServer((req, res) => {
    // Enable CORS
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
        res.writeHead(204);
        res.end();
        return;
    }

    // Serve index.html
    if (req.url === "/" || req.url === "/index.html") {
        const filePath = path.join(__dirname, "index.html");
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(500, { "Content-Type": "text/plain" });
                res.end("Internal Server Error");
                return;
            }
            res.writeHead(200, { "Content-Type": "text/html" });
            res.end(data);
        });
        return;
    }

    // Start Python script
    if (req.url === "/start") {
        if (!pythonProcess) {
            pythonProcess = spawn("python", [
                "AI_DRIVER_MONITORING.py"
            ], {
                shell: true,
                detached: true, // Allows the process to continue after the parent exits
                stdio: 'ignore' // Prevents hanging when parent exits
            });

            pythonProcess.unref(); // Allow parent to exit independently

            pythonProcess.on("error", (err) => {
                console.error("Failed to start Python process:", err);
                pythonProcess = null;
            });

            pythonProcess.on("exit", (code) => {
                console.log(`Python process exited with code ${code}`);
                pythonProcess = null;
            });

            res.writeHead(200, { "Content-Type": "text/plain" });
            res.end("Monitoring Started!");
        } else {
            res.writeHead(200, { "Content-Type": "text/plain" });
            res.end("Already Running!");
        }
        return;
    }

    // Stop Python script
    if (req.url === "/stop") {
        if (pythonProcess) {
            // Kill the entire process tree
            spawn("taskkill", ["/pid", pythonProcess.pid, "/f", "/t"]);
            pythonProcess = null;
            res.writeHead(200, { "Content-Type": "text/plain" });
            res.end("Monitoring Stopped!");
        } else {
            res.writeHead(200, { "Content-Type": "text/plain" });
            res.end("Not Running!");
        }
        return;
    }

    // Handle 404 Not Found
    res.writeHead(404, { "Content-Type": "text/plain" });
    res.end("Not Found");
});

const PORT = 5000;
server.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});

// Cleanup on server shutdown
process.on('SIGINT', () => {
    if (pythonProcess) {
        spawn("taskkill", ["/pid", pythonProcess.pid, "/f", "/t"]);
    }
    process.exit();
});