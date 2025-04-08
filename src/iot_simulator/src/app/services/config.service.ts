import { Injectable } from "@angular/core";
import { dump, load } from "js-yaml";
import { BehaviorSubject } from "rxjs";

/**
 * Clase que permite gestionar los estados de la aplicación.
 */
class StateManager {
    /** Número máximo de estados a almacenar. */
    private static readonly _MAX_STATES: number = 20;
    /** Lista de estados almacenados. */
    private _states: any[];
    /** Índice del estado actual. */
    private _index: number;
    /** Observable para emitir el estado actual y que el NetworkManageService pueda suscribirse. */
    private readonly _stateSubject: BehaviorSubject<any>;
    /** Estado actual. */
    public get state(): any {
        return this._states[this._index];
    }
    /** Mostrar el estado actual como un observable. */
    public get state$() {
        return this._stateSubject.asObservable();
    }
    /** Permite obtener si se puede deshacer un cambio. */
    public get canUndo(): boolean {
        return this._index > 0;
    }
    /** Permite obtener si se puede rehacer un cambio. */
    public get canRedo(): boolean {
        return this._index < this._states.length - 1;
    }

    /**
     * Constructor del gestor de estados.
     */
    public constructor() {
        const state = sessionStorage.getItem("state");

        if (state) this._states = [JSON.parse(state)];
        else this._states = [{}];
        this._index = 0;
        this._stateSubject = new BehaviorSubject<any>(
            this._states[this._index],
        );
    }

    /**
     * Manda un nuevo estado al observable.
     *
     * @param notify Notificar al suscriptor del estado actual.
     */
    private _stateChange(notify: boolean) {
        if (notify) this._stateSubject.next(this.state);
        sessionStorage.setItem("state", JSON.stringify(this.state));
    }

    /**
     * Establece un nuevo estado y lo limita a cierto número de elementos.
     *
     * @param state Nuevo estado a almacenar.
     */
    public setState(state: any, notify: boolean = true) {
        // Eliminar estados futuros
        this._states = this._states.slice(0, this._index + 1);
        // Agregamos el nuevo estado
        this._states.push(state);
        this._index++;
        // Eliminamos el más antiguo en caso de que se exceda el límite
        if (this._states.length > StateManager._MAX_STATES) {
            this._states.shift();
            this._index--;
        }
        // Notificamos el cambio
        this._stateChange(notify);
    }

    public replaceState(state: any, notify: boolean = true) {
        // Reemplazamos el estado actual
        this._states[this._index] = state;
        // Notificamos el cambio
        this._stateChange(notify);
    }

    /**
     * Deshace el último cambio.
     */
    public undo(notify: boolean = true) {
        if (this.canUndo) {
            // Decrementamos el índice
            this._index--;
            // Notificamos el cambio
            this._stateChange(notify);
        }
    }

    /**
     * Rehace el último cambio
     */
    public redo(notify: boolean = true) {
        if (this.canRedo) {
            // Incrementamos el índice
            this._index++;
            // Notificamos el cambio
            this._stateChange(notify);
        }
    }

    /**
     * Reinicia el gestor de estados.
     */
    public reset(notify: boolean = true) {
        // Limpiamos la lista de estados y el índice
        this._states = [{}];
        this._index = 0;
        // Notificamos el cambio
        this._stateChange(notify);
    }
}

@Injectable({ providedIn: "root" })
export class ConfigService {
    /** Gestor de estados de la aplicación. */
    public readonly stateManager: StateManager = new StateManager();

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
