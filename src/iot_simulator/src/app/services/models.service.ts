import { Injectable } from "@angular/core";
import { TensorflowModels } from "@models/cyber-shield";
import { ConfigService } from "@services/config.service";
import { io, LayersModel, loadLayersModel } from "@tensorflow/tfjs";
import { toast } from "ngx-sonner";
import { BehaviorSubject, Observable } from "rxjs";

/**
 * Clase que permite gestionar los modelos de inteligencia artificial de la aplicación.
 */
@Injectable()
export class ModelsService {
    /** Instancia única de la clase. */
    public static readonly instance: ModelsService = new ModelsService();
    /** Modelos cargados. */
    private _models?: TensorflowModels;
    /** Observable para emitir los modelos cargados. */
    private readonly _modelsSubject: BehaviorSubject<
        TensorflowModels | undefined
    >;
    /** Modelos cargados. */
    public get models(): TensorflowModels | undefined {
        return this._models ? this._models : undefined;
    }
    /** Mostrar los modelos como un observable. */
    public get models$(): Observable<TensorflowModels | undefined> {
        return this._modelsSubject.asObservable();
    }

    /** Constructor privado para evitar instanciación externa */
    private constructor() {
        // TODO: Cargar los modelos desde el localStorage
        this._modelsSubject = new BehaviorSubject<TensorflowModels | undefined>(
            this._models,
        );
    }

    /**
     * Carga los modelos desde una lista de archivos.
     *
     * @param files Lista de archivos a cargar.
     */
    private async _load(files: File[]): Promise<TensorflowModels | undefined> {
        const models: TensorflowModels = [];
        const topologyiesFiles: File[] = files.filter((file) =>
            file.name.endsWith(".json"),
        );
        const weightsFiles: File[] = files.filter((file) =>
            file.name.endsWith(".bin"),
        );
        // Generar los modelos
        for (const file of topologyiesFiles) {
            const model: LayersModel = await loadLayersModel(
                io.browserFiles([file, ...weightsFiles]),
            );
            const id = file.name.replace(/\.json$/, "");
            const name = id.replace(/_/g, " ");
            models.push({ id, name, model });
        }
        return models.length > 0 ? models : undefined;
    }

    /**
     * Cargar los modelos desde archivos.
     */
    public loadFromFile(): void {
        toast.promise(
            ConfigService.openFile(".json,.bin", true).then(this._load),
            {
                loading: "Importando modelos...",
                success: (models?: TensorflowModels) => {
                    this._models = models;
                    this._modelsSubject.next(this._models);
                    return "Modelos importados correctamente.";
                },
                error: (err) => "No se ha podido importar los modelos.",
            },
        );
    }
}
