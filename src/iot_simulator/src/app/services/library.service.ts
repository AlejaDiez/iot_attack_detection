import { Injectable } from "@angular/core";
import { ConfigService } from "@services/config.service";
import { parseScript } from "@utils/parse_script";
import { toast } from "ngx-sonner";
import { BehaviorSubject, Observable } from "rxjs";

/**
 * Clase que permite gestionar las bibliotecas de la aplicación.
 */
@Injectable()
export class LibraryService {
    /** Instancia única de la clase. */
    public static readonly instance: LibraryService = new LibraryService();
    /** Bibliotecas almacenada. */
    private _library?: Function = undefined;
    /** Biblioteca almacenada. */
    public get library(): Function | undefined {
        return this._library;
    }
    /** Observable para emitir el estado actual de la biblioteca cargada. */
    private readonly _librarySubject: BehaviorSubject<Function | undefined> =
        new BehaviorSubject<Function | undefined>(undefined);
    /** Observable que notifica cada vez que se carga una biblioteca */
    public get library$(): Observable<Function | undefined> {
        return this._librarySubject.asObservable();
    }

    /**
     * Constructor del gestor de bibliotecas.
     */
    private constructor() {
        const script = localStorage.getItem("script");

        if (script) this._load(script);
    }

    /**
     * Carga una biblioteca externa desde un string.
     *
     * @param script Contenido de la biblioteca a cargar.
     */
    private _load(script: string) {
        this._library = parseScript(script);
        this._librarySubject.next(this._library);
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
                return "Biblioteca importada correctamente.";
            },
            error: () => "No se ha podido importar la biblioteca.",
        });
    }
}
