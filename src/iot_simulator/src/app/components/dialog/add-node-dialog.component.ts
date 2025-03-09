import { Component, OnInit } from "@angular/core";
import {
    FormControl,
    FormGroup,
    ReactiveFormsModule,
    Validators,
} from "@angular/forms";
import { NodeType } from "@models/node";
import { BrnDialogRef, injectBrnDialogContext } from "@spartan-ng/brain/dialog";
import { BrnSelectImports } from "@spartan-ng/brain/select";
import { HlmButtonDirective } from "@spartan-ng/ui-button-helm";
import {
    HlmDialogFooterComponent,
    HlmDialogHeaderComponent,
    HlmDialogTitleDirective,
} from "@spartan-ng/ui-dialog-helm";
import { HlmInputDirective } from "@spartan-ng/ui-input-helm";
import { HlmLabelDirective } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";

export interface AddNodeDialogContext {
    name: string;
    type: NodeType;
}

@Component({
    selector: "app-add-node-dialog",
    imports: [
        BrnSelectImports,
        ReactiveFormsModule,
        HlmButtonDirective,
        HlmDialogHeaderComponent,
        HlmDialogTitleDirective,
        HlmDialogFooterComponent,
        HlmInputDirective,
        HlmLabelDirective,
        HlmSelectModule,
    ],
    template: `
        <hlm-dialog-header class="w-96">
            <h3 hlmDialogTitle>Añadir un nuevo {{ title }}</h3>
        </hlm-dialog-header>
        <form [formGroup]="form" (submit)="submit()">
            <div class="grid gap-4 py-4">
                <div class="grid grid-cols-4 items-center gap-4">
                    <label hlmLabel class="text-right">Nombre</label>
                    <input hlmInput formControlName="name" class="col-span-3" />
                </div>
                <div class="grid grid-cols-4 items-center gap-4">
                    <label hlmLabel class="text-right">Tipo</label>
                    <brn-select
                        [placeholder]="type"
                        formControlName="type"
                        class="col-span-3 inline-block">
                        <hlm-select-trigger class="w-full">
                            <hlm-select-value />
                        </hlm-select-trigger>
                        <hlm-select-content>
                            @for (item of types; track $index) {
                                <hlm-option [value]="item.value">{{
                                    item.label
                                }}</hlm-option>
                            }
                        </hlm-select-content>
                    </brn-select>
                </div>
            </div>
            <hlm-dialog-footer>
                <button hlmBtn type="submit">Añadir</button>
            </hlm-dialog-footer>
        </form>
    `,
    host: { class: "flex flex-col gap-4" },
})
//
export class AddNodeDialogComponent implements OnInit {
    private readonly _context!: AddNodeDialogContext;
    protected form: FormGroup = new FormGroup({
        name: new FormControl("", [Validators.required]),
        type: new FormControl(null, [Validators.required]),
    });
    protected get title(): string {
        if (this._context.type === NodeType.ROUTER) return "router";
        return "dispositivo";
    }
    protected get type(): string {
        switch (this.form.get("type")?.value) {
            case NodeType.ROUTER:
                return "Router";
            case NodeType.COMPUTER:
                return "Ordenador";
            case NodeType.IOT:
                return "Dispositivo IoT";
            default:
                return "Seleccione un tipo";
        }
    }
    protected get types(): { value: string; label: string }[] {
        if (this._context.type === NodeType.ROUTER)
            return [
                {
                    value: NodeType.ROUTER,
                    label: "Router",
                },
            ];
        return [
            {
                value: NodeType.COMPUTER,
                label: "Ordenador",
            },
            {
                value: NodeType.IOT,
                label: "Dispositivo IoT",
            },
        ];
    }

    constructor(private _ref: BrnDialogRef<AddNodeDialogContext>) {
        this._context = injectBrnDialogContext<AddNodeDialogContext>();
    }

    public ngOnInit(): void {
        this.form.get("type")!.setValue(this._context.type ?? null);
        if (this._context.type === NodeType.ROUTER)
            this.form.get("type")!.disable();
    }

    protected submit() {
        if (this.form.invalid) return;

        this._ref.close(this.form.getRawValue());
    }
}
