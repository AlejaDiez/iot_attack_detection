import { Component, inject, OnInit } from "@angular/core";
import {
    FormControl,
    FormGroup,
    ReactiveFormsModule,
    Validators,
} from "@angular/forms";
import { HlmSwitchModule } from "@components/ui/ui-switch-helm/src";
import { Connection } from "@models/connection";
import { NetworkService } from "@services/network.service";
import { StateService } from "@services/state.service";
import { BrnDialogRef, injectBrnDialogContext } from "@spartan-ng/brain/dialog";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import {
    HlmDialogHeaderComponent,
    HlmDialogTitleDirective,
} from "@spartan-ng/ui-dialog-helm";
import { HlmInputDirective } from "@spartan-ng/ui-input-helm";
import { HlmLabelDirective } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { latencyValidator } from "@validators/latency.validator";
import { isEqual } from "lodash";
import { debounceTime, distinctUntilChanged, filter } from "rxjs";

export interface EditConnectionContext {
    connection: Connection;
}

@Component({
    selector: "app-edit-connection-dialog",
    imports: [
        ReactiveFormsModule,
        BrnSelectModule,
        HlmDialogHeaderComponent,
        HlmDialogTitleDirective,
        HlmInputDirective,
        HlmLabelDirective,
        HlmSelectModule,
        HlmSwitchModule,
    ],
    template: `
        <hlm-dialog-header class="w-96">
            <h3 hlmDialogTitle>Modificar conexión</h3>
        </hlm-dialog-header>
        <form [formGroup]="form" class="grid gap-4 py-4">
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="text-right">Latencia</label>
                <input
                    hlmInput
                    type="number"
                    formControlName="latency"
                    class="col-span-3" />
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="text-right">Variabilidad</label>
                <input
                    hlmInput
                    type="number"
                    formControlName="latencyVariation"
                    class="col-span-3" />
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label hlmLabel class="text-right">Cyber Shield</label>
                <div class="col-span-3 inline-flex items-center justify-end">
                    <hlm-switch formControlName="cyberShield" />
                </div>
            </div>
            @if (form.get("cyberShield")!.value) {
                <div class="grid grid-cols-4 items-center gap-4">
                    <label hlmLabel class="text-right">Modelo</label>
                    <brn-select
                        placeholder="Selecciona un modelo"
                        formControlName="cyberShieldModel"
                        class="col-span-3 inline-block w-full">
                        <hlm-select-trigger class="w-full">
                            <hlm-select-value />
                        </hlm-select-trigger>
                        <hlm-select-content>
                            @for (
                                item of context.connection.cyberShield
                                    .availableModels;
                                track $index
                            ) {
                                <hlm-option [value]="item.id">
                                    {{ item.name }}
                                </hlm-option>
                            } @empty {
                                <hlm-option disabled class="px-2">
                                    No hay modelos disponibles
                                </hlm-option>
                            }
                        </hlm-select-content>
                    </brn-select>
                </div>
            }
        </form>
    `,
    host: { class: "flex flex-col gap-4" },
})
export class EditConnectionDialogComponent implements OnInit {
    protected readonly ref: BrnDialogRef<EditConnectionContext> = inject(
        BrnDialogRef<EditConnectionContext>,
    );
    protected readonly context =
        injectBrnDialogContext<EditConnectionContext>();
    protected readonly network: NetworkService = inject(NetworkService);
    protected readonly form: FormGroup = new FormGroup(
        {
            latency: new FormControl(null, {
                validators: [Validators.required, Validators.min(0)],
            }),
            latencyVariation: new FormControl(null, [Validators.min(0)]),
            cyberShield: new FormControl(false),
            cyberShieldModel: new FormControl(null),
        },
        [latencyValidator()],
    );

    public ngOnInit(): void {
        this.form.patchValue(
            {
                latency: this.context.connection.latency,
                latencyVariation: this.context.connection.latencyVariation,
                cyberShield: this.context.connection.cyberShield.enabled,
                cyberShieldModel:
                    this.context.connection.cyberShield.model?.id ?? null,
            },
            { emitEvent: false },
        );
        this.form.valueChanges
            .pipe(filter(() => this.form.valid))
            .subscribe((value) => {
                this.context.connection.latency = value.latency;
                this.context.connection.latencyVariation =
                    value.latencyVariation;
                this.context.connection.cyberShield.enabled = value.cyberShield;
                this.context.connection.cyberShield.model =
                    value.cyberShieldModel;
            });
        // Guardar el estado de la red
        this.form.valueChanges
            .pipe(
                filter(() => this.form.valid),
                debounceTime(StateService.UPDATE_WAIT),
                distinctUntilChanged((prev, curr) => isEqual(prev, curr)),
            )
            .subscribe(() => this.network.saveState());
    }
}
