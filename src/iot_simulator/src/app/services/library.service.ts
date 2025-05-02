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
    private static _instance: LibraryService;
    /** Instancia única de la clase. */
    public static get instance(): LibraryService {
        return LibraryService._instance;
    }
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
    private constructor(readonly config: ConfigService) {
        const script = localStorage.getItem("script");

        if (script) this._load(script);
    }

    /**
     * Método estático para inicializar la clase.
     */
    public static init(config: ConfigService): LibraryService {
        if (!LibraryService.instance)
            LibraryService._instance = new LibraryService(config);
        return LibraryService.instance;
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
            loading: this.config.translate.instant("LIBRARY_LOADING"),
            success: (script: string) => {
                this._load(script);
                return this.config.translate.instant("LIBRARY_LOADED");
            },
            error: () => this.config.translate.instant("LIBRARY_NOT_LOADED"),
        });
    }

    /**
     * Elimina la biblioteca externa cargada.
     */
    public deleteFile(): void {
        localStorage.removeItem("script");
        this._library = undefined;
        this._librarySubject.next(this._library);
        toast.success(this.config.translate.instant("LIBRARY_DELETED"));
    }
}
