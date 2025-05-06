import { Component, computed, inject, OnInit, Signal } from "@angular/core";
import {
    FormControl,
    FormGroup,
    ReactiveFormsModule,
    Validators,
} from "@angular/forms";
import { NodeType } from "@models/node";
import { TranslateModule } from "@ngx-translate/core";
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
        TranslateModule,
    ],
    template: `
        <hlm-dialog-header>
            <h3 hlmDialogTitle>{{ "ADD_NODE" | translate }}</h3>
        </hlm-dialog-header>
        <form [formGroup]="form" (submit)="submit()">
            <div class="grid gap-4 py-4">
                <div class="grid grid-cols-4 items-center gap-4">
                    <label hlmLabel class="text-right">{{
                        "NAME" | translate
                    }}</label>
                    <input hlmInput formControlName="name" class="col-span-3" />
                </div>
                <div class="grid grid-cols-4 items-center gap-4">
                    <label hlmLabel class="text-right">{{
                        "TYPE" | translate
                    }}</label>
                    <brn-select
                        [placeholder]="'SELECT_DEVICE_TYPE' | translate"
                        formControlName="type"
                        class="col-span-3 inline-block">
                        <hlm-select-trigger class="w-full">
                            <hlm-select-value />
                        </hlm-select-trigger>
                        <hlm-select-content>
                            @for (item of type(); track $index) {
                                <hlm-option [value]="item">{{
                                    NodeType.toString(item) | translate
                                }}</hlm-option>
                            }
                        </hlm-select-content>
                    </brn-select>
                </div>
            </div>
            <hlm-dialog-footer>
                <button hlmBtn [disabled]="form.invalid">
                    {{ "ADD" | translate }}
                </button>
            </hlm-dialog-footer>
        </form>
    `,
    host: { class: "min-w-96 flex flex-col gap-4" },
})
export class AddNodeDialogComponent implements OnInit {
    public readonly NodeType: typeof NodeType = NodeType;
    protected readonly ref: BrnDialogRef<AddNodeDialogContext> = inject(
        BrnDialogRef<AddNodeDialogContext>,
    );
    protected readonly context = injectBrnDialogContext<AddNodeDialogContext>();
    protected readonly type: Signal<NodeType[]> = computed(() =>
        Array.isArray(this.context.type)
            ? this.context.type
            : [this.context.type],
    );
    protected readonly form: FormGroup = new FormGroup({
        name: new FormControl(null, [Validators.required]),
        type: new FormControl(null, [Validators.required]),
    });

    public ngOnInit(): void {
        if (this.type().length === 1) {
            this.form.get("type")!.setValue(this.type()[0]);
            this.form.get("type")!.disable();
        }
    }

    protected submit() {
        this.ref.close(this.form.getRawValue());
    }
}
