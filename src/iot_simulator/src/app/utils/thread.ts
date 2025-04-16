export const newThread = async (fn: any, ...args: any[]) => {
    // Devuelve una promesa que se resuelve cuando el hilo termina
    // Se puede llamar a un metodo terminate para terminar el hilo
};
export class Thread {
    public readonly worker: Worker;

    constructor(fn: any) {
        const blob = new Blob(
            [
                `
  
          `,
            ],
            { type: "application/javascript" },
        );
        this.worker = new Worker(URL.createObjectURL(blob), {
            type: "module",
        });
    }

    public terminate() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
    }

    public run(...args: any[]): Promise<any> {
        return new Promise((resolve, reject) => {
            // Enviamos la función y los argumentos al Worker
            this.worker.postMessage({ fnStr: fn.toString(), args });

            // Manejamos la respuesta del Worker
            this.worker.onmessage = (event) => {
                const data = event.data;
                if (data.type === "resolve") {
                    this._resolve(data.result);
                } else if (data.type === "reject") {
                    this._reject(new Error(data.error));
                }
            };

            // Manejamos los errores del Worker
            this.worker.onerror = (event) => {
                this._reject(new Error(event.message));
            };
        });
    }
}

onmessage = function ({ fnStr, args }) {
    const fn = new Function("return " + fnStr)();

    try {
        const result = fn(...args);

        if (result && typeof result.then === "function")
            result.then(
                (res) => postMessage({ type: "resolve", result: res }),
                (err) => postMessage({ type: "reject", error: err }),
            );
        else postMessage({ type: "resolve", result });
    } catch (error) postMessage({ type: "reject", error: error.message });
};