import { Injectable } from "@angular/core";
import { parseScript } from "@utils/parse_script";
import { toast } from "ngx-sonner";
import { BehaviorSubject, Observable } from "rxjs";
import { ConfigService } from "./config.service";

/**
 * Clase que permite gestionar las bibliotecas de la aplicación.
 */
@Injectable({ providedIn: "root" })
export class LibraryService {
    /** Bibliotecas almacenada. */
    private _library?: Function;
    /** Observable para emitir el estado actual de la biblioteca cargada. */
    private readonly _librarySubject: BehaviorSubject<Function | undefined>;
    /** Estado actual de la biblioteca. */
    public get library(): Function | undefined {
        return this._library;
    }
    /** Mostrar el estado actual como un observable. */
    public get library$(): Observable<Function | undefined> {
        return this._librarySubject.asObservable();
    }

    /**
     * Constructor del gestor de bibliotecas.
     */
    public constructor() {
        // Cargar la biblioteca desde el localStorage
        const script = localStorage.getItem("script");

        if (script) this._load(script);
        this._librarySubject = new BehaviorSubject<Function | undefined>(
            this._library,
        );
    }

    /**
     * Carga una biblioteca externa desde un string.
     *
     * @param script Contenido de la biblioteca a cargar.
     */
    private _load(script: string) {
        this._library = parseScript(script);
        localStorage.setItem("script", script);
    }

    /**
     * Carga una biblioteca externa desde un archivo.
     */
    public loadFromFile(): void {
        toast.promise(ConfigService.openFile(".js"), {
            loading: "Importando biblioteca...",
            success: (script: string) => {
                this._load(script);
                this._librarySubject.next(this._library);
                return "Biblioteca importada correctamente.";
            },
            error: () => "No se ha podido importar la biblioteca.",
        });
    }
}
