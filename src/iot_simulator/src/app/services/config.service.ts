import { Injectable } from "@angular/core";
import { dump, load } from "js-yaml";

@Injectable({ providedIn: "root" })
export class ConfigService {
    /**
     * Abre un archivo existente.
     *
     * @returns Archivo abierto.
     */
    public async openFile(): Promise<any> {
        const input: HTMLInputElement = document.createElement("input");

        input.style.display = "none";
        input.type = "file";
        input.accept = ".yaml";
        input.click();
        return new Promise<any>((resolve, reject) => {
            input.onchange = async (event: any) => {
                const file: File = event.target.files[0];

                // Esperamos un segundo para que se pueda cargar el archivo
                await new Promise((_) => setTimeout(_, 1000));
                // Cargamos el archivo
                if (file) {
                    const reader = new FileReader();

                    reader.onload = () =>
                        resolve(load(reader.result as string));
                    reader.readAsText(file);
                } else reject();
                input.value = "";
                input.remove();
            };
        });
    }

    /**
     * Guarda un archivo.
     *
     * @param obj Objeto a guardar.
     */
    public async saveFile(obj: any): Promise<void> {
        const fileName: string = [
            "iot",
            "simulator",
            ...new Date().toISOString().split(/T|\./g, 2),
        ]
            .join("_")
            .replace(/-|:/g, "")
            .concat(".yaml");
        const fileContent: string = dump(obj, {
            noCompatMode: true,
            forceQuotes: true,
        });
        const fileBlob: Blob = new Blob([fileContent], {
            type: "application/x-yaml",
        });
        const link: HTMLAnchorElement = document.createElement("a");

        link.style.display = "none";
        link.href = URL.createObjectURL(fileBlob);
        link.download = fileName;

        // Esperamos un segundo para que se pueda descargar el archivo
        await new Promise((_) => setTimeout(_, 1000));
        // Descargamos el archivo
        link.click();
        URL.revokeObjectURL(link.href);
        link.remove();
    }
}
